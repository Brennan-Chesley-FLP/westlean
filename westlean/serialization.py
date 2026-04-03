"""Pydantic serialization models for all template inference algorithms.

Each algorithm's inferred template has a corresponding Pydantic model that
handles conversion to/from JSON-friendly dicts.  The top-level
``TemplateModel`` discriminated union dispatches on the ``algorithm`` field.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / Empty
# ---------------------------------------------------------------------------


class EmptyTemplateModel(BaseModel):
    algorithm: Literal["empty"] = "empty"


# ---------------------------------------------------------------------------
# ExAlg (token-frequency, linearized template)
# ---------------------------------------------------------------------------


class ExalgTokenModel(BaseModel):
    kind: str
    tag: str
    attr_name: str
    value: str
    position_key: str
    context: str


class ExalgLiteralModel(BaseModel):
    type: Literal["literal"] = "literal"
    token: ExalgTokenModel


class ExalgVarModel(BaseModel):
    type: Literal["var"] = "var"
    name: str
    kind: str
    tag: str
    attr_name: str
    position_key: str
    always_has_value: bool


class ExalgSetModel(BaseModel):
    type: Literal["set"] = "set"
    body: list[ExalgElementModel]
    var_name: str


class ExalgOptionalModel(BaseModel):
    type: Literal["optional"] = "optional"
    elements: list[ExalgElementModel]


ExalgElementModel = Annotated[
    Union[ExalgLiteralModel, ExalgVarModel, ExalgSetModel, ExalgOptionalModel],
    Field(discriminator="type"),
]

# Rebuild models that reference the forward ref
ExalgSetModel.model_rebuild()
ExalgOptionalModel.model_rebuild()


def _exalg_to_internal(elements: list[Any]) -> list[Any]:
    """Convert list of ExAlg Pydantic models to internal types."""
    from westlean.algorithms.exalg import Literal as ELiteral
    from westlean.algorithms.exalg import Optional as EOptional
    from westlean.algorithms.exalg import Set as ESet
    from westlean.algorithms.exalg import Token as EToken
    from westlean.algorithms.exalg import Var as EVar

    result: list[Any] = []
    for elem in elements:
        if isinstance(elem, ExalgLiteralModel):
            t = elem.token
            result.append(
                ELiteral(
                    EToken(
                        t.kind,
                        t.tag,
                        t.attr_name,
                        t.value,
                        t.position_key,
                        t.context,
                    )
                )
            )
        elif isinstance(elem, ExalgVarModel):
            result.append(
                EVar(
                    elem.name,
                    elem.kind,
                    elem.tag,
                    elem.attr_name,
                    elem.position_key,
                    elem.always_has_value,
                )
            )
        elif isinstance(elem, ExalgSetModel):
            result.append(ESet(_exalg_to_internal(elem.body), elem.var_name))
        elif isinstance(elem, ExalgOptionalModel):
            result.append(EOptional(_exalg_to_internal(elem.elements)))
    return result


class ExalgTemplateModel(BaseModel):
    algorithm: Literal["exalg"] = "exalg"
    elements: list[ExalgElementModel]

    def to_internal(self):
        from westlean.algorithms.exalg import ExalgTemplate

        return ExalgTemplate(_exalg_to_internal(self.elements))


# ---------------------------------------------------------------------------
# Anti-Unification & FiVaTech (shared tree node model)
# ---------------------------------------------------------------------------


class SlotModel(BaseModel):
    is_fixed: bool
    value: str


class RepeatingRegionModel(BaseModel):
    after_backbone_pos: int
    child_template: TplNodeModel
    var_name: str


class OptionalRegionModel(BaseModel):
    after_backbone_pos: int
    children: list[TplNodeModel]


class TplNodeModel(BaseModel):
    tag: str
    attr_names: list[str]
    text: SlotModel
    tail: SlotModel
    attrs: dict[str, SlotModel]
    children: list[TplNodeModel] | None
    children_var: str | None
    text_always_present: bool
    tail_always_present: bool
    # Multi-region fields (v2)
    repeating_regions: list[RepeatingRegionModel] = []
    optional_regions: list[OptionalRegionModel] = []
    # Legacy single-region fields (backward compat)
    repeating_child: TplNodeModel | None = None
    repeating_var: str | None = None
    optional_children: list[TplNodeModel] | None = None

    def _get_repeating_regions(self) -> list[RepeatingRegionModel]:
        """Get repeating regions, migrating from legacy fields if needed."""
        if self.repeating_regions:
            return self.repeating_regions
        if self.repeating_child is not None and self.repeating_var is not None:
            return [
                RepeatingRegionModel(
                    after_backbone_pos=-1,
                    child_template=self.repeating_child,
                    var_name=self.repeating_var,
                )
            ]
        return []

    def _get_optional_regions(self) -> list[OptionalRegionModel]:
        """Get optional regions, migrating from legacy fields if needed."""
        if self.optional_regions:
            return self.optional_regions
        if self.optional_children:
            return [
                OptionalRegionModel(
                    after_backbone_pos=-1,
                    children=self.optional_children,
                )
            ]
        return []

    def to_tpl_node(self):
        """Convert to anti_unification._TplNode."""
        from westlean.algorithms.anti_unification import _TplNode

        return _TplNode(
            tag=self.tag,
            attr_names=tuple(self.attr_names),
            text=(self.text.is_fixed, self.text.value),
            tail=(self.tail.is_fixed, self.tail.value),
            attrs={k: (v.is_fixed, v.value) for k, v in self.attrs.items()},
            children=(
                [c.to_tpl_node() for c in self.children]
                if self.children is not None
                else None
            ),
            children_var=self.children_var,
            text_always_present=self.text_always_present,
            tail_always_present=self.tail_always_present,
            repeating_regions=[
                (r.after_backbone_pos, r.child_template.to_tpl_node(), r.var_name)
                for r in self._get_repeating_regions()
            ],
            optional_regions=[
                (r.after_backbone_pos, [c.to_tpl_node() for c in r.children])
                for r in self._get_optional_regions()
            ],
        )

    def to_pattern_node(self):
        """Convert to fivatech._PatternNode."""
        from westlean.algorithms.fivatech import _PatternNode

        return _PatternNode(
            tag=self.tag,
            attr_names=tuple(self.attr_names),
            text=(self.text.is_fixed, self.text.value),
            tail=(self.tail.is_fixed, self.tail.value),
            attrs={k: (v.is_fixed, v.value) for k, v in self.attrs.items()},
            children=(
                [c.to_pattern_node() for c in self.children]
                if self.children is not None
                else None
            ),
            children_var=self.children_var,
            text_always_present=self.text_always_present,
            tail_always_present=self.tail_always_present,
            repeating_regions=[
                (r.after_backbone_pos, r.child_template.to_pattern_node(), r.var_name)
                for r in self._get_repeating_regions()
            ],
            optional_regions=[
                (r.after_backbone_pos, [c.to_pattern_node() for c in r.children])
                for r in self._get_optional_regions()
            ],
        )

    def to_uta_node(self):
        """Convert to tree_automata._UTANode."""
        from westlean.algorithms.tree_automata import _UTANode

        return _UTANode(
            tag=self.tag,
            attr_names=tuple(self.attr_names),
            text=(self.text.is_fixed, self.text.value),
            tail=(self.tail.is_fixed, self.tail.value),
            attrs={k: (v.is_fixed, v.value) for k, v in self.attrs.items()},
            children=(
                [c.to_uta_node() for c in self.children]
                if self.children is not None
                else None
            ),
            children_var=self.children_var,
            text_always_present=self.text_always_present,
            tail_always_present=self.tail_always_present,
            repeating_regions=[
                (r.after_backbone_pos, r.child_template.to_uta_node(), r.var_name)
                for r in self._get_repeating_regions()
            ],
            optional_regions=[
                (r.after_backbone_pos, [c.to_uta_node() for c in r.children])
                for r in self._get_optional_regions()
            ],
        )


class AntiUnifiedTemplateModel(BaseModel):
    algorithm: Literal["anti_unification"] = "anti_unification"
    root: TplNodeModel

    def to_internal(self):
        from westlean.algorithms.anti_unification import AntiUnifiedTemplate

        return AntiUnifiedTemplate(self.root.to_tpl_node())


class FiVaTechTemplateModel(BaseModel):
    algorithm: Literal["fivatech"] = "fivatech"
    root: TplNodeModel

    def to_internal(self):
        from westlean.algorithms.fivatech import FiVaTechTemplate

        return FiVaTechTemplate(self.root.to_pattern_node())


# ---------------------------------------------------------------------------
# RoadRunner
# ---------------------------------------------------------------------------


class TokenModel(BaseModel):
    kind: str
    tag: str
    attr_name: str
    value: str
    position_key: str


class LiteralModel(BaseModel):
    type: Literal["literal"] = "literal"
    token: TokenModel


class VarModel(BaseModel):
    type: Literal["var"] = "var"
    name: str
    kind: str
    tag: str
    attr_name: str
    position_key: str
    always_has_value: bool


class OptionalModel(BaseModel):
    type: Literal["optional"] = "optional"
    elements: list[UFREElementModel]


class RepeatModel(BaseModel):
    type: Literal["repeat"] = "repeat"
    elements: list[UFREElementModel]
    var_name: str


UFREElementModel = Annotated[
    Union[LiteralModel, VarModel, OptionalModel, RepeatModel],
    Field(discriminator="type"),
]

# Rebuild models that reference the forward ref
OptionalModel.model_rebuild()
RepeatModel.model_rebuild()


def _ufre_to_internal(elements: list[Any]) -> list[Any]:
    """Convert list of UFRE Pydantic models to internal types."""
    from westlean.algorithms.roadrunner import Literal as RRLiteral
    from westlean.algorithms.roadrunner import Optional as RROptional
    from westlean.algorithms.roadrunner import Repeat as RRRepeat
    from westlean.algorithms.roadrunner import Token, Var

    result: list[Any] = []
    for elem in elements:
        if isinstance(elem, LiteralModel):
            t = elem.token
            result.append(
                RRLiteral(Token(t.kind, t.tag, t.attr_name, t.value, t.position_key))
            )
        elif isinstance(elem, VarModel):
            result.append(
                Var(
                    elem.name,
                    elem.kind,
                    elem.tag,
                    elem.attr_name,
                    elem.position_key,
                    elem.always_has_value,
                )
            )
        elif isinstance(elem, OptionalModel):
            result.append(RROptional(_ufre_to_internal(elem.elements)))
        elif isinstance(elem, RepeatModel):
            result.append(RRRepeat(_ufre_to_internal(elem.elements), elem.var_name))
    return result


class RoadRunnerTemplateModel(BaseModel):
    algorithm: Literal["roadrunner"] = "roadrunner"
    ufre: list[UFREElementModel]

    def to_internal(self):
        from westlean.algorithms.roadrunner import RoadRunnerTemplate

        return RoadRunnerTemplate(_ufre_to_internal(self.ufre))


# ---------------------------------------------------------------------------
# k-Testable Tree Automata (UTA-based)
# ---------------------------------------------------------------------------


class KTestableTemplateModel(BaseModel):
    """Serialization model for the UTA-based k-Testable template.

    Uses the same ``TplNodeModel`` tree as Anti-Unification and FiVaTech.
    """

    algorithm: Literal["k_testable"] = "k_testable"
    k: int
    root: TplNodeModel

    def to_internal(self):
        from westlean.algorithms.tree_automata import KTestableTemplate

        return KTestableTemplate(self.root.to_uta_node(), self.k)


# ---------------------------------------------------------------------------
# Top-level discriminated union
# ---------------------------------------------------------------------------

TemplateModel = Annotated[
    Union[
        EmptyTemplateModel,
        ExalgTemplateModel,
        AntiUnifiedTemplateModel,
        FiVaTechTemplateModel,
        RoadRunnerTemplateModel,
        KTestableTemplateModel,
    ],
    Field(discriminator="algorithm"),
]


def _slot_model(slot: tuple[bool, str]) -> SlotModel:
    return SlotModel(is_fixed=slot[0], value=slot[1])


def tpl_node_to_model(node) -> TplNodeModel:
    """Convert a _TplNode or _PatternNode to TplNodeModel."""
    # Multi-region fields (new format)
    repeating_regions = []
    for pos, tpl, var in getattr(node, "repeating_regions", []):
        repeating_regions.append(
            RepeatingRegionModel(
                after_backbone_pos=pos,
                child_template=tpl_node_to_model(tpl),
                var_name=var,
            )
        )
    optional_regions = []
    for pos, children in getattr(node, "optional_regions", []):
        optional_regions.append(
            OptionalRegionModel(
                after_backbone_pos=pos,
                children=[tpl_node_to_model(c) for c in children],
            )
        )

    return TplNodeModel(
        tag=node.tag,
        attr_names=list(node.attr_names),
        text=_slot_model(node.text),
        tail=_slot_model(node.tail),
        attrs={k: _slot_model(v) for k, v in node.attrs.items()},
        children=(
            [tpl_node_to_model(c) for c in node.children]
            if node.children is not None
            else None
        ),
        children_var=node.children_var,
        text_always_present=node.text_always_present,
        tail_always_present=node.tail_always_present,
        repeating_regions=repeating_regions,
        optional_regions=optional_regions,
    )


def exalg_elements_to_model(
    elements: list,
) -> list[ExalgLiteralModel | ExalgVarModel | ExalgSetModel | ExalgOptionalModel]:
    """Convert internal ExAlg template elements to Pydantic models."""
    from westlean.algorithms.exalg import Literal as ELiteral
    from westlean.algorithms.exalg import Optional as EOptional
    from westlean.algorithms.exalg import Set as ESet
    from westlean.algorithms.exalg import Var as EVar

    result: list[
        ExalgLiteralModel | ExalgVarModel | ExalgSetModel | ExalgOptionalModel
    ] = []
    for elem in elements:
        if isinstance(elem, ELiteral):
            t = elem.token
            result.append(
                ExalgLiteralModel(
                    token=ExalgTokenModel(
                        kind=t.kind,
                        tag=t.tag,
                        attr_name=t.attr_name,
                        value=t.value,
                        position_key=t.position_key,
                        context=t.context,
                    ),
                )
            )
        elif isinstance(elem, EVar):
            result.append(
                ExalgVarModel(
                    name=elem.name,
                    kind=elem.kind,
                    tag=elem.tag,
                    attr_name=elem.attr_name,
                    position_key=elem.position_key,
                    always_has_value=elem.always_has_value,
                )
            )
        elif isinstance(elem, ESet):
            result.append(
                ExalgSetModel(
                    body=exalg_elements_to_model(elem.body),
                    var_name=elem.var_name,
                )
            )
        elif isinstance(elem, EOptional):
            result.append(
                ExalgOptionalModel(
                    elements=exalg_elements_to_model(elem.elements),
                )
            )
    return result


def ufre_to_model(
    elements: list,
) -> list[LiteralModel | VarModel | OptionalModel | RepeatModel]:
    """Convert internal UFRE elements to Pydantic models."""
    from westlean.algorithms.roadrunner import Literal as RRLiteral
    from westlean.algorithms.roadrunner import Optional as RROptional
    from westlean.algorithms.roadrunner import Repeat as RRRepeat
    from westlean.algorithms.roadrunner import Var

    result: list[LiteralModel | VarModel | OptionalModel | RepeatModel] = []
    for elem in elements:
        if isinstance(elem, RRLiteral):
            t = elem.token
            result.append(
                LiteralModel(
                    token=TokenModel(
                        kind=t.kind,
                        tag=t.tag,
                        attr_name=t.attr_name,
                        value=t.value,
                        position_key=t.position_key,
                    )
                )
            )
        elif isinstance(elem, Var):
            result.append(
                VarModel(
                    name=elem.name,
                    kind=elem.kind,
                    tag=elem.tag,
                    attr_name=elem.attr_name,
                    position_key=elem.position_key,
                    always_has_value=elem.always_has_value,
                )
            )
        elif isinstance(elem, RROptional):
            result.append(OptionalModel(elements=ufre_to_model(elem.elements)))
        elif isinstance(elem, RRRepeat):
            result.append(
                RepeatModel(
                    elements=ufre_to_model(elem.elements), var_name=elem.var_name
                )
            )
    return result
