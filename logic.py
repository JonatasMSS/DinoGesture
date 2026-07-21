"""Base de conhecimento proposicional e inferência por encadeamento para frente."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ACTIONS = frozenset({"pular", "abaixar", "neutro"})
ACTION_PREFIX = "acao:"
PERCEPT_COMBINATIONS = (
    frozenset({"boca_aberta", "sobrancelhas_levantadas"}),
    frozenset({"boca_aberta", "sobrancelhas_nao_levantadas"}),
    frozenset({"boca_fechada", "sobrancelhas_levantadas"}),
    frozenset({"boca_fechada", "sobrancelhas_nao_levantadas"}),
)


@dataclass(frozen=True)
class Rule:
    premises: frozenset[str]
    conclusion: str


class KnowledgeBase:
    """Armazena sentenças proposicionais e deriva suas consequências."""

    def __init__(self, rules: Iterable[Rule]) -> None:
        self.rules = tuple(rules)
        self.facts: set[str] = set()

    def tell(self, facts: Iterable[str]) -> None:
        self.facts.update(facts)

    def infer(self) -> set[str]:
        closure = set(self.facts)
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                if rule.premises <= closure and rule.conclusion not in closure:
                    closure.add(rule.conclusion)
                    changed = True
        return closure

    def ask(self, query: str) -> bool:
        return query in self.infer()

    def ask_one(self, candidates: Iterable[str]) -> str:
        closure = self.infer()
        answers = [candidate for candidate in candidates if candidate in closure]
        if len(answers) != 1:
            raise ValueError(f"A KB deve inferir exatamente uma ação, obteve: {answers}.")
        return answers[0]


@dataclass(frozen=True)
class KnowledgeDefinition:
    mouth_threshold: float
    brow_threshold: float
    rules: tuple[Rule, ...]

    def new_kb(self) -> KnowledgeBase:
        return KnowledgeBase(self.rules)

    def infer_action(self, facts: Iterable[str]) -> str:
        knowledge_base = self.new_kb()
        knowledge_base.tell(facts)
        action_atom = knowledge_base.ask_one(
            f"{ACTION_PREFIX}{action}" for action in ACTIONS
        )
        return action_atom.removeprefix(ACTION_PREFIX)


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} deve ser um número positivo.")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} deve ser um número positivo.")
    return value


def _rule(value: object, index: int) -> Rule:
    if not isinstance(value, dict):
        raise ValueError(f"Regra {index} deve ser um objeto JSON.")
    premises, conclusion = value.get("if"), value.get("then")
    if (
        not isinstance(premises, list)
        or not premises
        or not all(isinstance(atom, str) and atom for atom in premises)
        or not isinstance(conclusion, str)
        or not conclusion
    ):
        raise ValueError(f"Regra {index} deve conter 'if' não vazio e 'then'.")
    if conclusion.startswith(ACTION_PREFIX) and conclusion.removeprefix(ACTION_PREFIX) not in ACTIONS:
        raise ValueError(f"Ação desconhecida na regra {index}: {conclusion}.")
    return Rule(frozenset(premises), conclusion)


def load_knowledge(path: Path) -> KnowledgeDefinition:
    """Carrega e valida a KB antes de qualquer acesso à câmera ou ao Chrome."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Arquivo de regras não encontrado: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON de regras inválido: {error.msg}.") from error

    if not isinstance(data, dict):
        raise ValueError("A KB deve ser um objeto JSON.")
    rules_data = data.get("rules")
    if not isinstance(rules_data, list) or not rules_data:
        raise ValueError("A KB deve conter uma lista não vazia de regras.")
    definition = KnowledgeDefinition(
        mouth_threshold=_positive_number(data.get("mouth_threshold"), "mouth_threshold"),
        brow_threshold=_positive_number(data.get("brow_threshold"), "brow_threshold"),
        rules=tuple(_rule(rule, index) for index, rule in enumerate(rules_data, start=1)),
    )
    for facts in PERCEPT_COMBINATIONS:
        definition.infer_action(facts)
    return definition
