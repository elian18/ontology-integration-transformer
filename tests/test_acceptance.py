"""S7-T09: the fixed rule that accepts the concepts the AI extracts from a law (decision D1),
and its calibration against the decisions of Sprint 5 on the 93 concepts of the LOPDP."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.core import acceptance as ac
from src.core.acceptance import (ACCEPTED, REJECTED, REPEATED, agreement, apply_rule, check,
                                 label_key, summarize)


def _c(name="DataSubject", label="titular", definition="Persona dueña de los datos personales.",
       articles=(4,), ctype="class"):
    return {"name": name, "label": label, "definition": definition, "articles": list(articles),
            "type": ctype}


def _d(decision, same_as=None):
    return SimpleNamespace(decision=decision, same_as=same_as)


# ------------------------------------------------------------------ the rule

def test_label_key_ignores_case_accents_and_punctuation():
    assert label_key("Autoridad de  Protección, de Datos!") == "autoridad de proteccion de datos"
    assert label_key("Registro Nacional de protección") == label_key("registro nacional de "
                                                                     "Protección")
    assert label_key(None) == ""


def test_a_good_concept_passes():
    assert check(_c(), {4, 5}) == [] and check(_c()) == []


@pytest.mark.parametrize("change,reason", [
    ({"articles": []}, "no_article"),
    ({"articles": [4, 99]}, "unknown_article"),
    ({"name": "Data Subject"}, "bad_name"),
    ({"name": "Ab"}, "bad_name"),
    ({"name": "A" * 61}, "bad_name"),
    ({"name": "1Data"}, "bad_name"),
    ({"name": "data_subject"}, "bad_name"),
    ({"label": " "}, "no_label"),
    ({"label": "x" * 151}, "long_label"),
    ({"label": "dos\nlíneas"}, "long_label"),
    ({"definition": ""}, "no_definition"),
    ({"definition": "x" * 19}, "short_definition"),
    ({"definition": "x" * 601}, "long_definition"),
])
def test_each_reason(change, reason):
    assert check(_c(**change), {4, 5}) == [reason]


def test_the_limits_themselves_pass():
    ok = _c(name="Abc", label="x" * 150, definition="x" * 20)
    assert check(ok) == [] and check(_c(name="A" * 60, definition="x" * 600)) == []
    assert check({"name": None, "label": None, "definition": None, "articles": None}) == [
        "no_article", "bad_name", "no_label", "no_definition"]


def test_repeated_labels_join_the_first_one():
    entries = apply_rule([
        _c("DataSubject", "Titular", articles=[4]),
        _c("Data Owner", "titular"),                                   # rejected: bad name
        _c("DataOwner", "titular", articles=[33, 4]),
        _c("hasRecipient", "tiene destinatario", articles=[5], ctype="class"),
        _c("Owner", "TITULAR", articles=[7]),
    ], {4, 5, 7, 33})
    first, bad, owner, prop, again = entries
    assert first["status"] == ACCEPTED and first["joined"] == ["DataOwner", "Owner"]
    assert first["articles"] == [4, 33, 7]                     # their articles join the first
    assert bad["status"] == REJECTED and bad["reasons"] == ["bad_name"] and bad["kind"] is None
    assert (owner["status"], owner["repeated_of"]) == (REPEATED, "DataSubject")
    assert owner["articles"] == [33, 4]
    assert prop["kind"] == "property" and prop["declared_type"] == "class"   # from the name
    assert again["status"] == REPEATED
    assert first["kind"] == "class" and first["label"] == "Titular"


def test_a_class_and_a_property_are_never_the_same_concept():
    """S7-T09 on the GDPR: the property hasController was labelled "controller"."""
    entries = apply_rule([_c("Controller", "controller"), _c("hasController", "controller"),
                          _c("isControlledBy", "Controller")])
    assert [e["status"] for e in entries] == [ACCEPTED, ACCEPTED, REPEATED]
    assert entries[2]["repeated_of"] == "hasController" and entries[0]["joined"] == []


def test_a_rejected_concept_does_not_block_a_later_one():
    entries = apply_rule([_c("DataSubject", articles=[]), _c("Holder")])
    assert [e["status"] for e in entries] == [REJECTED, ACCEPTED]


def test_summarize():
    entries = apply_rule([_c("DataSubject"), _c("DataOwner"), _c("hasPurpose", "finalidad"),
                          _c("hasAim", "Finalidad"), _c("Bad", "malo", definition="corta")])
    assert summarize(entries) == {"concepts": 5, "accepted": 2, "repeated": 2, "rejected": 1,
                                  "classes": 1, "properties": 1,
                                  "reasons": {"short_definition": 1}}


# ------------------------------------------------------------------ calibration

def test_agreement_with_the_person():
    entries = apply_rule([_c("DataSubject"), _c("DataOwner"), _c("Holder"),
                          _c("Registry", "registro"), _c("NationalRegistry", "Registro"),
                          _c("Consent", "consentimiento"), _c("NoArticle", "x", articles=[]),
                          _c("Rejected", "rechazado", articles=[]), _c("Undecided", "nadie"),
                          _c("Approved", "aprobado"), _c("AlsoApproved", "Aprobado")])
    decisions = {"ai:DataSubject": _d("duplicate", "o#Owner"),
                 "ai:DataOwner": _d("duplicate", "o#Owner"),        # same entity: agrees
                 "ai:Holder": _d("duplicate", "o#Other"),           # another entity: disagrees
                 "ai:Registry": _d("approved"),
                 "ai:NationalRegistry": _d("rejected"),            # rejected: agrees
                 "ai:Consent": _d("rejected"),                     # accepted but rejected
                 "ai:NoArticle": _d("approved"),                   # rejected but kept
                 "ai:Rejected": _d("rejected"),
                 "ai:Approved": _d("approved"),
                 "ai:AlsoApproved": _d("approved")}                # both new: disagrees
    a = agreement(entries, decisions)
    assert (a["decided"], a["agree"]) == (10, 6) and a["agreement"] == 0.6
    assert a["accepted_but_rejected"] == ["Consent"]
    assert a["rejected_but_kept"] == ["NoArticle"]
    assert a["repeated_disagree"] == ["Holder", "AlsoApproved"]
    assert agreement([], {}) == {"decided": 0, "agree": 0, "agreement": None,
                                 "accepted_but_rejected": [], "rejected_but_kept": [],
                                 "repeated_disagree": [], "rows": []}


def test_a_repeated_concept_of_an_undecided_first_one_disagrees():
    entries = apply_rule([_c("DataSubject"), _c("DataOwner")])
    a = agreement(entries, {"ai:DataOwner": _d("duplicate", "o#Owner")})
    assert a["repeated_disagree"] == ["DataOwner"]


def _base(tmp_path, concepts, decisions):
    from src.alignment.decisions import TARGET_CONCEPT, append_decision, make_decision
    out, log = tmp_path / "out", tmp_path / "review" / "log.jsonl"
    out.mkdir()
    (out / "profile-proposed-concepts.json").write_text(json.dumps({"concepts": concepts}),
                                                        encoding="utf-8")
    for key, (decision, extra) in decisions.items():
        append_decision(make_decision(source_id="ontopriv+lopdp", target=TARGET_CONCEPT,
                                      concept_key=key, decision=decision, reviewer="Elian",
                                      **extra), log)
    return {"outputs": {"dir": str(out)}, "review": {"decisions_file": str(log)}}


def test_calibrate_cli_and_console(tmp_path, monkeypatch, capsys):
    concepts = [_c("DataSubject", articles=[4]), _c("DataOwner", articles=[33]),
                _c("Registry", "registro", articles=[]), _c("Consent", "consentimiento")]
    cfg = _base(tmp_path, concepts, {
        "ai:DataSubject": ("duplicate", {"same_as": "o#Data_Owner"}),
        "ai:DataOwner": ("duplicate", {"same_as": "o#Data_Owner"}),
        "ai:Registry": ("approved", {"entity_kind": "class"}),
        "ai:Consent": ("rejected", {})})
    result = ac.calibrate(cfg)
    assert result["rule_version"] == ac.RULE_VERSION and result["rule"] == ac.RULE_TEXT
    assert result["summary"]["accepted"] == 2 and result["agreement"]["agree"] == 2
    text = ac.render_calibration(result)
    assert text.splitlines()[0] == "Regla de aceptación de conceptos (S7-T09)"
    assert "Coincide con tus decisiones del Sprint 5 en 2 de 4 (50,0 %)" in text
    assert ('    repetido: DataOwner se une a DataSubject ("titular"); lo confirmaste como '
            'Data_Owner') in text
    assert ('    aceptado por la regla pero lo rechazaste: Consent ("consentimiento")') in text
    assert ("    rechazado por la regla (no cita un artículo de la ley) pero lo conservaste: "
            "Registry") in text
    monkeypatch.setattr("src.config.load_config", lambda: cfg)
    assert ac._run([]) == 0
    saved = json.loads((tmp_path / "out" / ac.CALIBRATION_FILE).read_text(encoding="utf-8"))
    assert saved["agreement"]["decided"] == 4
    assert "Guardado en:" in capsys.readouterr().out
    (tmp_path / "out" / "profile-proposed-concepts.json").unlink()
    assert ac._run([]) == 1
    with pytest.raises(FileNotFoundError, match="conceptos de la IA"):
        ac.calibrate(cfg)


def test_repeated_rejected_and_disagreeing_lines():
    summary = {"concepts": 2, "accepted": 1, "repeated": 1, "rejected": 0}
    result = {"rule": "r", "summary": summary,
              "agreement": {"agree": 1, "decided": 2, "agreement": 0.5,
                            "accepted_but_rejected": [], "rejected_but_kept": [],
                            "repeated_disagree": ["B"],
                            "rows": [{"name": "B", "label": "b", "rule": REPEATED,
                                      "repeated_of": "A", "person": "rejected",
                                      "same_as": None, "reasons": []}]}}
    text = ac.render_calibration(result)
    assert '    repetido: B se une a A ("b"); lo rechazaste' in text
    assert "    repetido según la regla pero lo decidiste distinto: B" in text
    assert ac._pct(None) == "—"


def test_real_calibration_on_the_lopdp():
    """The rule on the 93 concepts of Sprint 3 against the decisions of Sprint 5."""
    files = ac.base_files()
    if not files["proposals"].exists() or not files["log"].exists():
        pytest.skip("Faltan los conceptos de la IA o el registro de decisiones de la base")
    result = ac.calibrate()
    s, a = result["summary"], result["agreement"]
    assert (s["concepts"], s["accepted"], s["repeated"], s["rejected"]) == (93, 87, 5, 1)
    assert (a["decided"], a["agree"]) == (93, 91)
    assert a["accepted_but_rejected"] == ["ExplicitConsent"]
    assert a["rejected_but_kept"] == ["UnfulfilledRegistry"]
    assert a["repeated_disagree"] == []
    repeated = {r["name"]: r["repeated_of"] for r in a["rows"] if r["rule"] == REPEATED}
    assert repeated["DataProtectionRegistry"] == "NationalDataProtectionRegistry"
    assert repeated["DataOwner"] == "DataSubject"