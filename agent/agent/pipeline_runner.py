"""
agent.pipeline_runner: end to end document analysis pipeline.

    files -> OCR -> corpus -> Hodge chunk -> query -> LLM -> hallucination check -> rechunk

Every step calls compiled Cython kernels. This module is glue.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


#: the gate could not be evaluated, so the caller proceeds unannotated
GATE_OK = "ok"
#: measurable but weak coverage: answer, and say the context is thin
GATE_WARN = "warn"
#: the query and the source share no vocabulary, so nothing was retrieved
GATE_REFUSE = "refuse"


def _context_quality_gate(source_rex, source_labels, query) -> dict:
    """Structural coverage gate: does the source cover the query well enough to
    trust the retrieved context?

    Returns {"verdict", "reasons", "score", "n_shared"} where verdict is one of
    GATE_OK, GATE_WARN, GATE_REFUSE.

    REFUSE is reserved for a MEASURED zero: the query supplies terms and none of
    them name a vertex, so retrieval returned nothing and any answer would come
    from the model alone rather than the source. Being unable to evaluate is not
    the same finding and stays permissive.

    WARN is read off `quality_gate`, which is q(x) = x / (x + median(|x|)) per
    channel. That puts its own fixed point at 0.5, where a channel scores exactly
    the median magnitude, so the comparison is against the statistic the kernel
    already normalises by rather than against a tuned constant.

    The channel scored path consumes the library's native three score bundle.
    No spectral coverage, eigenbasis or legacy four component confidence reading
    is needed to evaluate this gate.
    """
    def verdict(v, *reasons, score=None, n_shared=None):
        return {"verdict": v, "reasons": list(reasons),
                "score": score, "n_shared": n_shared}

    if not source_rex or source_rex.nE == 0:
        return verdict(GATE_OK, "no source complex to measure against")
    try:
        # A token that names a vertex counts however short it is: identifiers are
        # routinely two characters, and dropping them by length would report zero
        # coverage for a query that names its subject exactly. The length rule only
        # decides which NON matching tokens are substantive enough to judge on.
        index_of = {str(lbl).lower(): i for i, lbl in enumerate(source_labels)}
        tokens = [w.lower().strip(".,;:!?()[]\"'") for w in query.split()]
        shared = [index_of[t] for t in tokens if t in index_of]
        usable = [t for t in tokens if len(t) > 2 or t in index_of]
        if not usable:
            return verdict(GATE_OK, "query carries no terms substantive enough to match")
        if not shared:
            return verdict(
                GATE_REFUSE,
                f"none of the {len(usable)} query terms name a vertex in the source",
                n_shared=0)

        # The library already builds this bundle and dispatches it: eigen free and
        # sparse at any size, via `sparse_interfacing`. Hand assembling it from the
        # dense kernel cost two things. It needed a full L0 eigenbasis, so the gate
        # silently reported "unavailable" on exactly the large complexes worth
        # gating; and it passed `g_channel_operator` for G where the library's own
        # interfacing path passes `L_overlap`, so this score was never the score the
        # rest of the platform computes.
        #
        # target_signal=None is the self interfacing reading, which is what scoring
        # psi against itself meant, at one L0^+ solve instead of two.
        from rexgraph.core._interfacing import quality_gate
        bundle = source_rex.interfacing_vector(
            np.array(shared, dtype=np.int32),
            np.ones(len(shared), dtype=np.float64),
            None,
        )
        ch_scores = np.asarray(bundle["scores"], dtype=np.float64)
        gate = quality_gate(ch_scores.reshape(1, -1))
        if not isinstance(gate, np.ndarray):
            return verdict(GATE_OK, "gate returned no score", n_shared=len(shared))
        score = float(gate.mean())
        if score <= 0.5:
            return verdict(GATE_WARN,
                           f"channel score {score:.3f} at or below the gate's median "
                           "fixed point",
                           score=score, n_shared=len(shared))
        return verdict(GATE_OK, score=score, n_shared=len(shared))
    except Exception as exc:
        return verdict(GATE_OK, f"gate could not be evaluated: {exc}")


@dataclass
class PipelineResult:
    """Complete pipeline output."""
    documents: list = field(default_factory=list)
    corpus_summary: dict = field(default_factory=dict)
    temporal: dict = field(default_factory=dict)
    chunks: list = field(default_factory=list)
    query_result: dict = field(default_factory=dict)
    model_response: str = ""
    hallucination_report: dict = field(default_factory=dict)
    ontology: dict = field(default_factory=dict)
    elapsed: float = 0.0


class PipelineRunner:
    """Runs the full OCR -> analysis -> LLM pipeline."""

    # "read" = load documents into the corpus: parse tables/JSON/text directly,
    # or OCR image/PDF files. Named domain agnostically (it is NOT always OCR).
    PHASES = ["read", "corpus", "analysis", "chunking", "query", "model", "hallucination"]

    OCR_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

    def __init__(self, ocr_client=None, model_url=None, max_vocab=200):
        self._ocr_client = ocr_client
        self._model_url = model_url
        self.max_vocab = max_vocab
        self._callbacks = []
        self._last_corpus = None
        self._stage_callback = None

    def on_phase(self, callback):
        """Register a callback for phase progress: callback(phase_name, data_dict)."""
        self._callbacks.append(callback)

    def _emit(self, phase, data):
        for cb in self._callbacks:
            with contextlib.suppress(Exception):
                cb(phase, data)

    def run(
        self,
        files: list[str] | None = None,
        texts: list[str] | None = None,
        doc_ids: list[str] | None = None,
        query: str | None = None,
        max_rechunk: int = 2,
        depth: str = "standard",
        ontology: bool = False,
        face_selection: str | None = None,
        reader_options: dict | None = None,
    ) -> PipelineResult:
        """Run the full pipeline.

        Pass ``files`` for auto detected processing, or ``texts``/``doc_ids``
        for pre extracted text (when OCR is cached in the caller).

        Files are routed by type via auto_rex:
            PDF/images  -> OCR -> text -> CorpusBuilder
            CSV/TSV     -> csv_loader -> RexGraph
            JSON        -> json_loader -> RexGraph
            .txt/other  -> read as text -> TextAdapter

        When ``ontology`` is True, an optional TrustGraph enrichment stage
        runs after chunking.
        """
        t0 = time.time()
        self._last_corpus = None
        if ((texts is None) != (doc_ids is None)
                or (texts is not None and len(texts) != len(doc_ids))):
            raise ValueError("texts and doc_ids must be supplied together with equal lengths")
        if doc_ids is not None and len(set(doc_ids)) != len(doc_ids):
            raise ValueError("doc_ids must be unique")
        result = PipelineResult()

        # Per analysis stage callback -> surfaced as "analysis" phase events
        # so the frontend can show real progress during the long
        # eigendecomposition/Hodge stages.
        def stage_cb(doc_id, stage_name, stage_data):
            payload = {"status": "running", "doc_id": doc_id, "stage": stage_name}
            if isinstance(stage_data, dict) and "error" in stage_data:
                payload["stage_error"] = stage_data["error"]
            self._emit("analysis", payload)

        self._stage_callback = stage_cb

        # Step 1: build one corpus from whatever inputs are present.
        # A mixed batch (OCR texts + direct CSV/JSON/text files) must
        # Include extracted text and directly readable source files.
        have_texts = texts is not None and doc_ids is not None
        have_files = bool(files)
        if not have_texts and not have_files:
            result.elapsed = time.time() - t0
            return result

        if have_texts:
            self._emit("read", {"status": "skipped", "n_texts": len(texts)})
        if have_files:
            self._emit("read", {"status": "running", "n_files": len(files)})
        corpus = self._build_corpus(
            texts if have_texts else None,
            doc_ids if have_texts else None,
            files if have_files else None,
            depth,
            face_selection=face_selection,
            reader_options=reader_options,
        )
        if have_files:
            self._emit("read", {"status": "done"})

        if corpus is None:
            result.elapsed = time.time() - t0
            return result

        # Step 2: extract doc info
        self._emit("corpus", {"status": "running", "n_docs": len(corpus.documents)})
        result.documents = self._extract_doc_info(corpus)
        result.corpus_summary = corpus.summary() if corpus._built else {}
        self._emit("corpus", {"status": "done", "n_documents": len(result.documents)})

        # Temporal
        try:
            result.temporal = corpus.temporal_tags()
        except Exception as e:
            logger.warning("Temporal tags failed: %s", e)

        # Step 3: chunk each document
        self._emit("chunking", {"status": "running"})
        all_chunks = self._chunk_documents(corpus)
        result.chunks = [
            {
                "doc_id": doc_id,
                "chunks": [
                    {
                        "idx": c.idx,
                        "text_preview": c.text[:150],
                        "char_start": c.char_start,
                        "char_end": c.char_end,
                        "n_edges": c.n_edges,
                        "kappa": c.kappa,
                        "hodge_gradient": c.hodge_gradient,
                        "hodge_curl": c.hodge_curl,
                        "hodge_harmonic": c.hodge_harmonic,
                        "dominant_channel": c.dominant_channel,
                    }
                    for c in chunks
                ],
            }
            for doc_id, chunks in all_chunks
        ]
        total_chunks = sum(len(dc["chunks"]) for dc in result.chunks)
        self._emit("chunking", {"status": "done", "total_chunks": total_chunks})

        # Optional TrustGraph ontology enrichment
        if ontology:
            self._emit("ontology", {"status": "running"})
            try:
                result.ontology = corpus.trustgraph_analysis(depth=depth)
                self._emit("ontology", {
                    "status": "done",
                    "available": result.ontology.get("available", False),
                    "n_triples": result.ontology.get("n_triples", 0),
                })
            except Exception as e:
                logger.warning("Ontology stage failed: %s", e)
                result.ontology = {"available": False, "reason": str(e)}
                self._emit("ontology", {"status": "error", "error": str(e)})

        # Step 4: query + LLM + hallucination check
        if query:
            self._emit("query", {"status": "running", "query": query})
            result.query_result, result.model_response, result.hallucination_report = \
                self._query_and_check(corpus, all_chunks, texts, query, max_rechunk)
            self._emit("query", {"status": "done"})

        result.elapsed = time.time() - t0
        return result

    def _ocr_files(self, files):
        """OCR each file, return (texts, doc_ids)."""
        from agent.integrations.unlimited_ocr import (
            create_ocr_client,
            is_image_file,
            is_pdf_file,
        )
        client = self._ocr_client or create_ocr_client()

        texts = []
        doc_ids = []
        for path in files:
            try:
                if is_pdf_file(path):
                    result = client.ocr_pdf(path)
                    text = result.full_text
                elif is_image_file(path):
                    result = client.ocr_image(path)
                    text = result.text
                else:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        text = f.read()

                if text and len(text.strip()) > 10:
                    texts.append(text)
                    doc_ids.append(Path(path).stem)
                else:
                    logger.warning("No text extracted from %s", path)
            except Exception as e:
                logger.warning("Failed to process %s: %s", path, e)

        return texts, doc_ids

    def _add_file_to_corpus(self, corpus, path):
        """Add one direct file to the corpus, routed by type.

        OCR extension files are OCR'd and routed through the OCRAdapter's
        structure aware layout path. Everything else is added
        as a source path so auto_rex picks the right adapter (CSV/JSON/
        feature/text): never flattened to word co occurrence.
        """
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix in self.OCR_EXTENSIONS:
            text = self._ocr_single_file(path)
            text = self._sanitize_text(text)
            if not text or not text.strip():
                raise ValueError(f"No readable text was extracted from {path}")
            ec = None
            try:
                from agent.adapters.ocr import OCRAdapter
                ec = OCRAdapter().build_from_text(text, strategy="layout")
                if ec is None or ec.nE == 0:
                    ec = None
            except Exception as e:
                logger.warning(
                    "OCRAdapter layout failed for %s (%s); "
                    "using flat text.", p.name, e)
                ec = None
            corpus.add_document(source=path, text=text, edge_construction=ec)
        else:
            corpus.add_document(source=path)

    def _build_corpus(self, texts, doc_ids, files, depth, *,
                      face_selection=None, reader_options=None):
        """Build a single corpus from OCR texts and/or direct files."""
        from agent.corpus import CorpusBuilder
        if reader_options is not None and not isinstance(reader_options, dict):
            raise TypeError("reader_options must be a dictionary")
        options = dict(reader_options or {})
        options.setdefault("max_vocab", self.max_vocab)
        if face_selection is not None:
            options["face_selection"] = face_selection
        corpus = CorpusBuilder(**options)
        if texts and doc_ids:
            for text, doc_id in zip(texts, doc_ids, strict=False):
                corpus.add_text(text, doc_id=doc_id)
        for path in (files or []):
            self._add_file_to_corpus(corpus, path)
        if not corpus.documents:
            return None
        corpus.build(depth=depth, stage_callback=self._stage_callback)
        failures = [f"{doc.doc_id}: {doc.meta.get('build_error', 'No document relations were constructed')}"
                    for doc in corpus.documents if doc.rex is None or doc.rex.nE == 0]
        if failures:
            raise ValueError("Document construction failed: " + "; ".join(failures))
        self._last_corpus = corpus
        return corpus

    def _build_corpus_from_files(self, files, depth):
        """Build corpus from file paths (thin wrapper over _build_corpus)."""
        return self._build_corpus(None, None, files, depth)

    def _build_corpus_from_texts(self, texts, doc_ids, depth):
        """Build corpus from pre extracted texts (thin wrapper)."""
        return self._build_corpus(texts, doc_ids, None, depth)

    def _ocr_single_file(self, path):
        """OCR a single PDF or image file."""
        from agent.integrations.unlimited_ocr import create_ocr_client, is_image_file, is_pdf_file
        client = self._ocr_client or create_ocr_client()
        try:
            if is_pdf_file(path):
                return client.ocr_pdf(path).full_text
            elif is_image_file(path):
                return client.ocr_image(path).text
        except Exception as e:
            raise RuntimeError(f"OCR failed for {path}: {e}") from e
        return ""

    @staticmethod
    def _sanitize_text(text):
        """Normalize OCR whitespace while preserving extracted content."""
        if not text:
            return text
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def _extract_doc_info(self, corpus):
        """Extract per document analysis info."""
        docs = []
        for doc in corpus.documents:
            d = {"doc_id": doc.doc_id}
            if doc.rex:
                d["nV"] = doc.rex.nV
                d["nE"] = doc.rex.nE
                d["nF"] = doc.rex.nF
                topo = doc.analysis.get("topology", {})
                d["betti"] = topo.get("betti", [])
                rel = doc.analysis.get("relational", {})
                d["kappa_mean"] = rel.get("kappa_mean")
                d["chi_mean"] = rel.get("chi_mean")
                hodge = doc.analysis.get("hodge", {})
                d["hodge"] = {
                    "gradient": hodge.get("pct_gradient"),
                    "curl": hodge.get("pct_curl"),
                    "harmonic": hodge.get("pct_harmonic"),
                }
            docs.append(d)
        return docs

    def _chunk_documents(self, corpus):
        """Hodge chunk each document."""
        from agent.chunking import hodge_chunk

        results = []
        for doc in corpus.documents:
            if doc.rex is None:
                continue
            if not doc.text or len(doc.text.strip()) < 50:
                continue
            try:
                metadata = getattr(doc.rex, "_agent_meta", {}) or {}
                from agent.adapters import EdgeSpan, SentenceSpan
                edges = metadata.get("edge_spans")
                sentences = metadata.get("sentence_spans")
                if edges is None or sentences is None:
                    from agent.chunking import Chunk
                    chunks = [Chunk(0, doc.text, 0, len(doc.text), 0, 0)]
                else:
                    chunks = hodge_chunk(doc.rex,
                        [EdgeSpan(**value) if isinstance(value, dict) else value for value in edges],
                        [SentenceSpan(**value) if isinstance(value, dict) else value for value in sentences], doc.text)
                results.append((doc.doc_id, chunks))
            except Exception as e:
                logger.warning("Chunking failed for %s: %s", doc.doc_id, e)

        return results

    def _query_and_check(self, corpus, all_chunks, texts, query, max_rechunk):
        """Query, generate and observe against the same emitted context selection."""
        from agent.hallucination import detect_hallucinations_exchange
        from agent.model_context import diagnostic_source, select_context
        qr = corpus.query(query, top_k=3, mode="hybrid")
        query_dict = {"query": qr.query_text, "ranked": qr.ranked_sections}
        by_id = {doc.doc_id: doc for doc in corpus.documents}
        chunk_groups = dict(all_chunks)
        entries = []
        for ranked in (qr.ranked_sections or [])[:2]:
            doc = by_id.get(ranked.get("doc_id"))
            if doc is None:
                continue
            chunks = chunk_groups.get(doc.doc_id, ())
            scores = np.asarray([chunk.kappa for chunk in chunks], dtype=float)
            finite = scores[np.isfinite(scores)]
            q1, q3 = np.percentile(finite, [25, 75]) if len(finite) >= 4 else (0, float("inf"))
            fence = float(q1-1.5*(q3-q1))
            entries.extend((doc, chunk) for chunk in chunks if np.isfinite(chunk.kappa) and chunk.kappa >= fence)
        selection = select_context(entries)
        context_rex, labels = diagnostic_source(selection)
        gate = _context_quality_gate(context_rex, labels, query)
        measured = gate["n_shared"] is not None
        gate_report = {"context_gate": gate["verdict"], "context_gate_reasons": gate["reasons"],
            "context_score": gate["score"],
            "context_evaluation": "evaluated" if measured else "not_evaluated",
            "context_sufficient": gate["verdict"] != GATE_REFUSE if measured else None,
            "context": selection.as_record(), "evaluation_status": "not_evaluated"}
        if gate["verdict"] == GATE_REFUSE:
            return query_dict, "The selected context does not cover this query: " + "; ".join(gate["reasons"]), gate_report
        try:
            response = self._call_model(query, selection.text)
        except Exception as exc:
            gate_report["model_error"] = f"{type(exc).__name__}: {exc}"
            return query_dict, f"Error: {exc}", gate_report
        report = detect_hallucinations_exchange(selection.text, response)
        return query_dict, response, {**gate_report, **report.as_record()}

    def _call_model(self, query, context):
        """Send query + context to the model."""
        import httpx

        url = self._model_url
        if not url:
            from agent.cli.serve import find_running_server
            url = find_running_server()
        if not url:
            url = os.environ.get("CHAT_MODEL_URL", "") or os.environ.get("UNLIMITED_OCR_URL", "")
        if not url:
            raise RuntimeError("No GPU server running")

        messages = []
        if context:
            messages.append({
                "role": "system",
                "content": f"Document context:\n{context}",
            })
        messages.append({"role": "user", "content": query})

        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{url}/v1/chat/completions",
                json={"messages": messages, "max_tokens": 1024, "temperature": 0.7},
            )
            resp.raise_for_status()
            data = resp.json()
            text = ""
            for choice in data.get("choices", []):
                text += choice.get("message", {}).get("content", "")
            return text
