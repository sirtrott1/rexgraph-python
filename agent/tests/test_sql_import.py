"""Optional live SQL sources convert complete data into primary native records."""
from contextlib import closing
from datetime import datetime,date,time,timezone
from decimal import Decimal
import json,struct,uuid
from pathlib import Path

import pytest
sa=pytest.importorskip('sqlalchemy')
from agent.sql_source import SQLSource,SQLDataset,import_sql,_encode,_decode
from agent.agentic_db import AgenticDB
from agent import rcdb
from agent.hive import Hive


@pytest.fixture
def source(tmp_path):
    from rexgraph.io.sql_bridge import get_engine
    uri='sqlite:///'+str(tmp_path/'source.db')
    engine=get_engine(uri)
    with engine.begin() as c:
        c.exec_driver_sql('CREATE TABLE parent (a INTEGER, b INTEGER, value TEXT, PRIMARY KEY (a,b))')
        c.exec_driver_sql('CREATE TABLE child (id INTEGER PRIMARY KEY, a INTEGER,b INTEGER, large NUMERIC,n INTEGER, data BLOB, FOREIGN KEY (a,b) REFERENCES parent(a,b))')
        c.exec_driver_sql('CREATE TABLE duplicates (value TEXT, empty INTEGER)')
        c.exec_driver_sql('CREATE TABLE empty_table (id INTEGER)')
        c.exec_driver_sql('INSERT INTO parent VALUES (1,2,\'α\'),(1,3,\'β\')')
        c.exec_driver_sql('INSERT INTO child VALUES (?,?,?,?,?,?)',[(1,1,2,2**60+1,None,b'\x00\xff'),(2,1,3,0,0,b'')])
        c.exec_driver_sql('INSERT INTO duplicates VALUES (?,?)',[('same',None),('same',None)])
    return uri,engine


def test_live_source_reads_and_hive_workers_preserve_catalog_and_values(source):
    uri,engine=source
    selected=SQLSource(uri)
    catalog=selected.catalog()
    child=next(t for t in catalog if t['table']=='child')
    assert child['foreign_keys'][0]['constrained_columns']==['a','b']
    assert child['foreign_keys'][0]['referred_columns']==['a','b']
    rows=selected.read('child',equals={'id':1})
    assert rows[0]['large']==2**60+1 and type(rows[0]['large']) is int
    assert rows[0]['n'] is None and rows[0]['data']==b'\x00\xff'
    hive=Hive('sql-test');selected.attach_to_hive(hive)
    assert hive.invoke('sql.read',{'table':'parent','equals':{'b':3}})==[{'a':1,'b':3,'value':'β'}]
    with pytest.raises(ValueError):selected.read('DROP TABLE parent')
    assert len(selected.read('parent'))==2


@pytest.mark.parametrize('value',[None,False,0,2**100+3,-0.0,float('inf'),Decimal('12345678901234567890.123400'),
    Decimal('-0.00'),date(2026,10,23),time(1,2,3),datetime(2026,10,23,1,2,tzinfo=timezone.utc),
    uuid.UUID(int=7),b'\0\xff',{'$sql_type':['decimal','not-a-tag']},(1,None)])
def test_sql_scalar_encoding_is_explicit_and_reversible(value):
    actual=_decode(_encode(value))
    assert type(actual) is type(value)
    if type(value) is float:assert struct.pack('>d',actual)==struct.pack('>d',value)
    elif isinstance(value,Decimal):assert actual.as_tuple()==value.as_tuple()
    else:assert actual==value


def test_complete_import_and_native_files_work_without_the_original_sql(source,tmp_path):
    uri,engine=source
    with AgenticDB('local://'+str(tmp_path/'records'),writable=True) as db:
        result=db.import_sql(uri,'dataset',batch_size=1)
        assert result['rows']==6 and result['tables']==4 and result['batches']==7
        dataset=db.sql_dataset('dataset')
        expected={name:SQLSource(uri).read(name) for name in ['parent','child','duplicates','empty_table']}
        for name,rows in expected.items():assert list(dataset.rows(name))==rows
        for format in ['rcbf','rcbd','safetensors','hdf5']:
            path=dataset.export(tmp_path/format,format=format)
            reader=SQLDataset.from_directory(path)
            for name,rows in expected.items():assert list(reader.rows(name))==rows
        manifest=dataset.manifest
        assert manifest['tables'][0]['declaration']['foreign_keys'][0]['constrained_columns']==['a','b']
        child=next(t for t in manifest['tables'] if t['declaration']['table']=='child')['batches'][0]
        value=db.query('FROM RCDB_VERSION($db,$id,1) RETURN COUNT(CELLS(1))',parameters={'id':child['record_id']})
        assert value.values==(1,)
    engine.dispose()
    with AgenticDB('local://'+str(tmp_path/'records')) as db:
        assert list(db.sql_dataset('dataset').rows('child'))==expected['child']


def test_failed_import_never_publishes_a_complete_manifest(source,tmp_path,monkeypatch):
    uri,_=source
    with closing(rcdb.LocalStore(tmp_path/'records')) as store:
        real=SQLSource.iter_table
        def failing(self,table,**kwargs):
            yield from real(self,table,**kwargs)
            raise RuntimeError('interrupted source')
        monkeypatch.setattr(SQLSource,'iter_table',failing)
        with pytest.raises(RuntimeError):import_sql(uri,store,'incomplete',batch_size=1)
        assert store.get_record('incomplete') is None
        assert store.list()


def test_destination_revision_guard_precedes_source_connection(source,tmp_path,monkeypatch):
    uri,_=source
    with closing(rcdb.LocalStore(tmp_path/'records')) as store:
        import_sql(uri,store,'dataset')
        def forbidden(self):pytest.fail('opened SQL after stale destination revision')
        monkeypatch.setattr(SQLSource,'__enter__',forbidden)
        with pytest.raises(rcdb.VersionConflictError):import_sql(uri,store,'dataset',expected_version=0)


@pytest.mark.parametrize('format',['rcbf','rcbd'])
def test_cli_command_imports_and_exports(source,tmp_path,capsys,monkeypatch,format):
    from agent.cli.connect import main
    uri,_=source;monkeypatch.setenv('SQL_IMPORT_TEST_URI',uri)
    result=main(['import-sql','--source-env','SQL_IMPORT_TEST_URI','--store','local://'+str(tmp_path/'records'),
                 '--id','data','--batch-size','1','--files',str(tmp_path/'export'),'--format',format,'--json'])
    assert result==0
    assert json.loads(capsys.readouterr().out)['rows']==6
    assert list(SQLDataset.from_directory(tmp_path/'export').rows('duplicates'))==[{'value':'same','empty':None}]*2


def test_one_sql_source_has_independent_worker_snapshots(source):
    from concurrent.futures import ThreadPoolExecutor
    uri,_=source
    selected=SQLSource(uri)
    with ThreadPoolExecutor(max_workers=4) as workers:
        rows=list(workers.map(lambda i:selected.read('parent',equals={'b':2+i%2}),range(12)))
    assert all(len(row)==1 for row in rows)
    assert [row[0]['b'] for row in rows]==[2+i%2 for i in range(12)]


def test_source_file_and_failed_destination_are_not_overwritten(source,tmp_path):
    import hashlib
    uri,engine=source
    path=Path(engine.url.database);before=hashlib.sha256(path.read_bytes()).hexdigest()
    with closing(rcdb.LocalStore(tmp_path/'records')) as store:
        import_sql(uri,store,'dataset')
        target=tmp_path/'existing';target.mkdir();(target/'keep').write_text('original')
        with pytest.raises(FileExistsError):SQLDataset(store,'dataset').export(target)
        assert (target/'keep').read_text()=='original'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before


def test_native_directory_can_be_converted_again_and_checks_batch_identity(source,tmp_path):
    uri,_=source
    with closing(rcdb.LocalStore(tmp_path/'records')) as store:
        import_sql(uri,store,'dataset')
        SQLDataset(store,'dataset').export(tmp_path/'first')
    dataset=SQLDataset.from_directory(tmp_path/'first')
    dataset.export(tmp_path/'second',format='safetensors')
    assert list(SQLDataset.from_directory(tmp_path/'second').rows('child'))==list(dataset.rows('child'))
    index=tmp_path/'first'/'EXPORT.json';contents=json.loads(index.read_text())
    contents['batches'][0]['file']=contents['batches'][1]['file']
    index.write_text(json.dumps(contents))
    with pytest.raises(ValueError,match='identity'):
        list(SQLDataset.from_directory(tmp_path/'first').rows('child'))


def test_missing_warehouse_constraint_catalogs_remain_unknown(source,tmp_path,monkeypatch):
    uri,_=source
    real=sa.inspect
    class PartialInspector:
        def __init__(self,connection):self.inner=real(connection)
        def __getattr__(self,name):return getattr(self.inner,name)
        def get_pk_constraint(self,*args,**kwargs):raise NotImplementedError
        def get_foreign_keys(self,*args,**kwargs):raise NotImplementedError
    monkeypatch.setattr(sa,'inspect',lambda connection:PartialInspector(connection))
    with closing(rcdb.LocalStore(tmp_path/'records')) as store:
        import_sql(uri,store,'dataset')
        dataset=SQLDataset(store,'dataset')
        declaration=dataset.manifest['tables'][0]['declaration']
        assert declaration['primary_key'] is None and declaration['foreign_keys'] is None
        assert {'primary_key','foreign_keys'}<=set(declaration['unavailable'])
        assert len(list(dataset.rows('child')))==2


def test_thread_local_in_memory_source_cannot_silently_change_databases():
    from concurrent.futures import ThreadPoolExecutor
    engine=sa.create_engine('sqlite://')
    with engine.begin() as connection:connection.exec_driver_sql('CREATE TABLE original(id INTEGER)')
    selected=SQLSource(engine)
    assert selected.catalog()[0]['table']=='original'
    with ThreadPoolExecutor(max_workers=1) as worker:
        with pytest.raises(RuntimeError,match='thread-local'):
            worker.submit(selected.catalog).result()
    engine.dispose()


def test_explicit_schema_count_weights_preserve_supported_zero(source):
    from agent.connectors.sql import SQLConnector
    uri,engine=source
    with engine.begin() as connection:connection.exec_driver_sql('DELETE FROM child')
    graph,metadata=SQLConnector(with_weights=True).read(uri)
    assert metadata['weights']==[0]
    assert graph.w_E.tolist()==[0.0]
    assert metadata['modality'][0]['from_columns']==['a','b']
    assert metadata['modality'][0]['to_columns']==['a','b']


def test_schema_view_does_not_glue_equal_table_names_from_different_schemas(source):
    from agent.sql_source import schema_model
    uri,_=source;catalog=SQLSource(uri).catalog()
    child=next(table for table in catalog if table['table']=='child')
    child['foreign_keys'][0]['referred_schema']='different_schema'
    assert schema_model(catalog).foreign_keys==[]
    assert child['foreign_keys'][0]['referred_columns']==['a','b']
