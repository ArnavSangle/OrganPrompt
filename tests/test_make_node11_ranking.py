from scripts.make_node11_ranking import panel_label


def test_panel_label_names_the_first_organ_and_its_distance():
    info = {"top_organ": "lung-right", "distance_mm": 27.3}
    assert panel_label(info) == "lung-right, 27 mm"
