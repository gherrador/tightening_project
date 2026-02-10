from tightening_project.data.io.naming import parse_source_stem

def test_parse_source_stem_ok():
    info = parse_source_stem("TighteningProduct_November25_260115114110_1")
    assert info.year == 2025
    assert info.month == 11
    assert info.export_ts == "260115114110"
    assert info.source_stem.startswith("TighteningProduct")
