"""Zoom-view axis limits and titles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewDefinition:
    viewId: str
    axisMinAu: float
    axisMaxAu: float
    titleFontSize: int
    shortName: str
    title: str
