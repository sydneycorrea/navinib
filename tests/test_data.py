from navinib import config
from navinib.data.generate import generate
from navinib.db import connect
from navinib.rag.corpus import CALL_NOTES


def test_world_size_and_rep(con):
    assert con.execute("SELECT COUNT(*) FROM provider").fetchone()[0] == config.N_PROVIDERS
    assert con.execute("SELECT COUNT(*) FROM rep").fetchone()[0] == 1


def test_call_note_providers_exist(con):
    ids = {r[0] for r in con.execute("SELECT provider_id FROM provider")}
    assert {pid for _, pid, *_ in CALL_NOTES} <= ids


def test_generation_is_reproducible(tmp_path):
    rows = []
    for name in ("a.db", "b.db"):
        c = connect(generate(tmp_path / name))
        rows.append(c.execute("SELECT * FROM provider ORDER BY provider_id").fetchall())
        c.close()
    assert [tuple(r) for r in rows[0]] == [tuple(r) for r in rows[1]]


def test_provider_ids_are_a_sequence(con):
    ids = [r[0] for r in con.execute("SELECT provider_id FROM provider ORDER BY provider_id")]
    assert ids == [f"p-{n:03d}" for n in range(1, config.N_PROVIDERS + 1)]


def test_no_drug_brand_names(con):
    names = " ".join(r[0] for r in con.execute("SELECT full_name || affiliation FROM provider JOIN location USING (provider_id)"))
    assert "Product X" not in names  # the only drug name in the project is the placeholder


def test_csv_export(tmp_path):
    from navinib.data.export_csv import export
    out = export(tmp_path / "csv", generate(tmp_path / "x.db"))
    assert {p.name for p in out.iterdir()} == {"providers.csv", "contacts.csv", "call_history.csv",
                                               "demo_day_schedule.csv", "call_notes.csv"}
