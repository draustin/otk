from otk.refindinf import lookup_index, search, _main


def test_main():
    pages = search('Eimerl-o', 'BaB2O4')
    assert len(pages) == 1
    entry = _main.load_page(pages[0])
    index = _main.parse_entry(entry)
    assert (abs(index(220e-9) - 1.8284) < 1e-3)

def test_lookup_index():
    assert (abs(lookup_index('Malitson', 'SiO2')(800e-9) - 1.4533) < 1e-3)
