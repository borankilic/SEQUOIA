from sequoia.gui.palette import PART_MIME, Palette


def top_names(palette):
    tree = palette.tree
    return [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]


def test_tree_content_and_order(qtbot, parts_library):
    palette = Palette(parts_library)
    qtbot.addWidget(palette)
    names = top_names(palette)
    assert names == [
        "Sources",
        "Freespace",
        "Filters",
        "Fiber",
        "Detectors",
        "Electronics",
        "Probes",
    ]
    assert palette.item_for_ref("sources/laser") is not None
    assert palette.item_for_ref("probes/powermeter") is not None
    probes = palette.item_for_ref("probes/powermeter").parent().parent()
    assert probes.text(0) == "Probes"


def test_problems_group(qtbot, parts_library):
    bad = parts_library.root / "misc" / "mystery"
    bad.mkdir(parents=True)
    (bad / "part.yaml").write_text("type: NoSuchType\nparams: {}\n")
    broken = parts_library.root / "misc" / "broken"
    broken.mkdir(parents=True)
    (broken / "part.yaml").write_text("- not a mapping\n")
    palette = Palette(parts_library)
    qtbot.addWidget(palette)
    assert top_names(palette)[-1] == "Problems"
    item = palette.item_for_ref("misc/mystery")
    assert item.parent().text(0) == "Problems"
    assert "NoSuchType" in item.toolTip(0)
    assert palette.item_for_ref("misc/broken").toolTip(0)


def test_search_filters_and_clears(qtbot, parts_library):
    palette = Palette(parts_library)
    qtbot.addWidget(palette)
    everything = palette.visible_refs()
    palette.search.setText("laser")
    assert palette.visible_refs() == ["sources/laser"]
    palette.search.setText("no such thing")
    assert palette.visible_refs() == []
    palette.search.setText("")
    assert palette.visible_refs() == everything


def test_mime_and_activation(qtbot, parts_library):
    palette = Palette(parts_library)
    qtbot.addWidget(palette)
    item = palette.item_for_ref("freespace/mirror")
    mime = palette.tree.mimeData([item])
    assert bytes(mime.data(PART_MIME)).decode() == "freespace/mirror"
    group = item.parent()
    assert not palette.tree.mimeData([group]).hasFormat(PART_MIME)
    seen = []
    palette.part_activated.connect(seen.append)
    palette._on_double_click(item, 0)
    palette._on_double_click(group, 0)
    assert seen == ["freespace/mirror"]


def test_refresh_picks_up_new_parts(qtbot, parts_library):
    palette = Palette(parts_library)
    qtbot.addWidget(palette)
    assert palette.item_for_ref("sources/laser_2") is None
    new = parts_library.root / "sources" / "laser_2"
    new.mkdir(parents=True)
    (new / "part.yaml").write_text(
        "type: DummyLaser\nmodel: Second\nparams:\n  wavelength: 780 nm\n"
    )
    palette.refresh()
    item = palette.item_for_ref("sources/laser_2")
    assert item is not None and item.text(0) == "Second"
