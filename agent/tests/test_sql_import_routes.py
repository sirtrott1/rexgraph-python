"""The SQL import button and tool use existing upload, workspace and audit gates."""
import io,zipfile
import pytest
sa=pytest.importorskip('sqlalchemy')
from fastapi.testclient import TestClient
from agent import rcdb
from agent.sql_source import SQLDataset
from agent.mcp_tools import Context,call


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('REXGRAPH_CONFIG_DIR',str(tmp_path/'config'))
    monkeypatch.setenv('REXGRAPH_RCDB_URI','local://'+str(tmp_path/'records'))
    monkeypatch.setenv('REXGRAPH_SECRETS_URI','file://'+str(tmp_path/'secrets.json'))
    from agent.server import auth,scope
    auth.reset_auth_manager();scope.reset_secret_store();rcdb.reset_default_store()
    auth.get_auth_manager().disable_auth(persist=False)
    from agent.server.app import app
    with TestClient(app) as selected:yield selected
    auth.reset_auth_manager();scope.reset_secret_store();rcdb.reset_default_store()


@pytest.mark.parametrize('format',['rcbf','rcbd'])
def test_uploaded_sqlite_file_imports_through_the_tool_boundary(client,tmp_path,format):
    path=tmp_path/'source.db';engine=sa.create_engine('sqlite:///'+str(path))
    with engine.begin() as connection:
        connection.exec_driver_sql('CREATE TABLE source(id INTEGER PRIMARY KEY,value TEXT)')
        connection.exec_driver_sql("INSERT INTO source VALUES (1,'alpha'),(2,'beta')")
    engine.dispose()
    upload=client.post('/rex/v1/upload',content=path.read_bytes(),headers={'X-Filename':'source.db'})
    handle=upload.json()['handle']
    catalog=client.post('/api/v1/dbmanager/sql/catalog',json={'files':[handle]})
    assert catalog.status_code==200,catalog.text
    assert catalog.json()['result']['tables'][0]['table']=='source'
    arguments={'files':[handle],'record_id':'sql-data','expected_version':0,'batch_size':1}
    result=client.post('/api/v1/dbmanager/sql/import',json=arguments)
    assert result.status_code==200,result.text
    assert result.json()['result']['rows']==2
    dataset=SQLDataset(rcdb.default_store(),'sql-data')
    assert list(dataset.rows('source'))==[{'id':1,'value':'alpha'},{'id':2,'value':'beta'}]
    assert client.post('/api/v1/dbmanager/sql/import',json=arguments).status_code==400
    exported=client.get('/api/v1/dbmanager/sql/export',params={'record_id':'sql-data','format':format})
    assert exported.status_code==200,exported.text
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        assert 'EXPORT.json' in archive.namelist()
        if format=='rcbd':
            assert any(name.startswith('dataset.rcbd/') for name in archive.namelist())
        else:
            assert 'dataset.rcbf' in archive.namelist()
        archive.extractall(tmp_path/'download')
    assert list(SQLDataset.from_directory(tmp_path/'download').rows('source'))==list(dataset.rows('source'))
    assert rcdb.default_store().backend=='local'


def test_sql_source_rejects_raw_server_file_names_at_socket_boundary(client,tmp_path):
    result=client.post('/api/v1/dbmanager/sql/catalog',json={'uri':'sqlite:///'+str(tmp_path/'private.db')})
    assert result.status_code==400 and 'upload' in result.text.lower()
    assert not (tmp_path/'private.db').exists()


def test_sql_server_source_policy_and_credentials_are_preserved(client,monkeypatch):
    monkeypatch.setenv('REXGRAPH_ALLOWED_DB_HOSTS','allowed.example')
    result=client.post('/api/v1/dbmanager/sql/catalog',json={'uri':'postgresql://user:secret@other.example/data'})
    assert result.status_code==400 and 'secret' not in result.text


def test_named_source_cannot_cross_explicit_tool_contexts(tmp_path,monkeypatch):
    monkeypatch.setenv('REXGRAPH_SECRETS_URI','file://'+str(tmp_path/'secrets.json'))
    from agent.secrets import open_secret_store
    from agent.server.scope import ScopedSecrets
    ScopedSecrets(open_secret_store(),'alpha').put('prod','postgresql://user:secret@host/database')
    with pytest.raises(KeyError):
        call('rexgraph_sql_catalog',context=Context(workspace='beta',identity='bob',auth_enabled=True),connection='prod')
