from westlean.content_model import ELEMENTS, children_allowed, get_valid_children


class TestContentModel:
    def test_void_elements_have_no_children(self):
        for tag, edef in ELEMENTS.items():
            if edef.void:
                assert get_valid_children(tag) == [], (
                    f"{tag} is void but has valid children"
                )

    def test_ul_only_accepts_li(self):
        children = get_valid_children("ul")
        assert children == ["li"]

    def test_ol_only_accepts_li(self):
        children = get_valid_children("ol")
        assert children == ["li"]

    def test_table_accepts_table_sections(self):
        children = set(get_valid_children("table"))
        assert children == {"caption", "thead", "tbody", "tfoot", "tr"}

    def test_tr_only_accepts_cells(self):
        children = set(get_valid_children("tr"))
        assert children == {"td", "th"}

    def test_p_accepts_phrasing_only(self):
        children = get_valid_children("p")
        for tag in children:
            assert ELEMENTS[tag].categories & ELEMENTS["span"].categories, (
                f"{tag} in <p> children but not phrasing"
            )

    def test_div_accepts_flow(self):
        children = get_valid_children("div")
        assert "div" in children
        assert "p" in children
        assert "span" in children
        assert "ul" in children

    def test_header_forbids_nested_header(self):
        assert not children_allowed(ELEMENTS["header"], "header")
        assert not children_allowed(ELEMENTS["header"], "footer")

    def test_a_forbids_nested_a(self):
        assert not children_allowed(ELEMENTS["a"], "a")

    def test_dl_accepts_dt_dd(self):
        children = set(get_valid_children("dl"))
        assert children == {"dt", "dd"}

    def test_select_accepts_option_optgroup(self):
        children = set(get_valid_children("select"))
        assert children == {"option", "optgroup"}

    def test_all_elements_registered(self):
        expected_tags = {
            "html",
            "head",
            "body",
            "title",
            "div",
            "section",
            "article",
            "aside",
            "nav",
            "header",
            "footer",
            "main",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "span",
            "a",
            "strong",
            "em",
            "code",
            "b",
            "i",
            "small",
            "mark",
            "time",
            "p",
            "blockquote",
            "pre",
            "ul",
            "ol",
            "li",
            "dl",
            "dt",
            "dd",
            "table",
            "caption",
            "thead",
            "tbody",
            "tfoot",
            "tr",
            "td",
            "th",
            "br",
            "hr",
            "img",
            "input",
            "form",
            "label",
            "select",
            "option",
            "textarea",
            "button",
        }
        for tag in expected_tags:
            assert tag in ELEMENTS, f"Missing element: {tag}"
