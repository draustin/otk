from otk import rii


def test_rii():
    pages = rii.search('Eimerl-o', 'BaB2O4')
    assert len(pages) == 1
    entry = rii.load_page(pages[0])
    index = rii.parse_entry(entry)
    assert (abs(index(220e-9) - 1.8284) < 1e-3)

def test_lookup_index():
    assert (abs(rii.lookup_index('Malitson', 'SiO2')(800e-9) - 1.4533) < 1e-3)
