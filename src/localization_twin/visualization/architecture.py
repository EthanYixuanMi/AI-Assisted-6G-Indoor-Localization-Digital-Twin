"""Dependency-free SVG export for the system-architecture reference figure."""

from __future__ import annotations

from html import escape
from pathlib import Path


ARCHITECTURE_COLUMNS = (
    (
        "Scenario Controller",
        ("YAML profiles", "Seeds & perturbations", "Replay selection"),
        "#E69F00",
    ),
    (
        "Environment & Signal Twin",
        ("Floor plan / anchors", "LoS & NLoS geometry", "RSS measurements"),
        "#56B4E9",
    ),
    (
        "Data & Feature Layer",
        ("Spatial splits", "Masks & distances", "Traceable artifacts"),
        "#0072B2",
    ),
    (
        "Localization Engines",
        ("Geometric LS / KNN", "Direct & residual AI", "Kalman filtering"),
        "#009E73",
    ),
    (
        "Evaluation & Dashboard",
        ("Accuracy & robustness", "Runtime / ablations", "Interactive replay"),
        "#CC79A7",
    ),
)


def export_system_architecture_svg(path: str | Path) -> Path:
    """Write an editable, accessible SVG architecture diagram."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    width = 1500
    height = 650
    margin = 55
    gap = 34
    node_width = (width - 2 * margin - gap * 4) / 5
    node_y = 210
    node_height = 275
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title description">'
        ),
        "<title id=\"title\">AI-Assisted 6G Indoor Localization Digital Twin architecture</title>",
        (
            "<desc id=\"description\">Five-stage flow from scenario control "
            "through the environment and signal twin, data features, localization "
            "engines, and evaluation dashboard.</desc>"
        ),
        "<defs>",
        (
            '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#516A7C"/></marker>'
        ),
        (
            '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">'
            '<feDropShadow dx="0" dy="7" stdDeviation="8" '
            'flood-color="#00101E" flood-opacity="0.18"/></filter>'
        ),
        "</defs>",
        '<rect width="1500" height="650" rx="24" fill="#F5F8FA"/>',
        (
            '<text x="55" y="72" font-family="Arial, Helvetica, sans-serif" '
            'font-size="31" font-weight="700" fill="#142636">'
            "AI-Assisted Indoor Localization Digital Twin</text>"
        ),
        (
            '<text x="55" y="109" font-family="Arial, Helvetica, sans-serif" '
            'font-size="17" fill="#52697B">'
            "Reproducible simulation-to-evidence workflow · CPU-only software proof of concept</text>"
        ),
        (
            '<rect x="55" y="137" width="1390" height="2" '
            'fill="#D2DEE7"/>'
        ),
    ]
    for index, (title, bullets, color) in enumerate(ARCHITECTURE_COLUMNS):
        x = margin + index * (node_width + gap)
        if index < len(ARCHITECTURE_COLUMNS) - 1:
            line_start = x + node_width + 7
            line_end = x + node_width + gap - 8
            parts.append(
                (
                    f'<line x1="{line_start:.1f}" y1="{node_y + node_height / 2:.1f}" '
                    f'x2="{line_end:.1f}" y2="{node_y + node_height / 2:.1f}" '
                    'stroke="#516A7C" stroke-width="3" marker-end="url(#arrow)"/>'
                )
            )
        parts.extend(
            [
                (
                    f'<g id="stage-{index + 1}" filter="url(#shadow)">'
                    f'<rect x="{x:.1f}" y="{node_y}" width="{node_width:.1f}" '
                    f'height="{node_height}" rx="17" fill="#FFFFFF" '
                    'stroke="#D1DEE7" stroke-width="2"/>'
                    f'<rect x="{x:.1f}" y="{node_y}" width="{node_width:.1f}" '
                    f'height="13" rx="7" fill="{color}"/>'
                    "</g>"
                ),
                (
                    f'<circle cx="{x + 31:.1f}" cy="{node_y + 53}" r="18" '
                    f'fill="{color}"/>'
                ),
                (
                    f'<text x="{x + 31:.1f}" y="{node_y + 60}" '
                    'font-family="Arial, Helvetica, sans-serif" font-size="19" '
                    'font-weight="700" text-anchor="middle" fill="#FFFFFF">'
                    f"{index + 1}</text>"
                ),
                (
                    f'<text x="{x + 58:.1f}" y="{node_y + 48}" '
                    'font-family="Arial, Helvetica, sans-serif" font-size="18" '
                    'font-weight="700" fill="#162938">'
                    f'<tspan x="{x + 58:.1f}" dy="0">{escape(title.split(" & ")[0])}</tspan>'
                    + (
                        f'<tspan x="{x + 58:.1f}" dy="23">&amp; '
                        f"{escape(title.split(' & ', 1)[1])}</tspan>"
                        if " & " in title
                        else ""
                    )
                    + "</text>"
                ),
            ]
        )
        bullet_y = node_y + 119
        for bullet_index, bullet in enumerate(bullets):
            y = bullet_y + bullet_index * 48
            parts.extend(
                [
                    (
                        f'<circle cx="{x + 29:.1f}" cy="{y - 5}" r="4" '
                        f'fill="{color}"/>'
                    ),
                    (
                        f'<text x="{x + 43:.1f}" y="{y}" '
                        'font-family="Arial, Helvetica, sans-serif" '
                        'font-size="15.5" fill="#415A6B">'
                        f"{escape(bullet)}</text>"
                    ),
                ]
            )
    parts.extend(
        [
            (
                '<path d="M 1337 518 C 1337 575, 170 575, 170 518" '
                'fill="none" stroke="#93A8B7" stroke-width="2" '
                'stroke-dasharray="7 7" marker-end="url(#arrow)"/>'
            ),
            (
                '<rect x="515" y="548" width="470" height="42" rx="21" '
                'fill="#E8EEF3" stroke="#CAD8E2"/>'
            ),
            (
                '<text x="750" y="575" text-anchor="middle" '
                'font-family="Arial, Helvetica, sans-serif" font-size="15" '
                'font-weight="600" fill="#40596B">'
                "Saved metrics and diagnostics inform the next scenario replay</text>"
            ),
            (
                '<text x="1445" y="625" text-anchor="end" '
                'font-family="Arial, Helvetica, sans-serif" font-size="12" '
                'fill="#718596">Simulation architecture reference</text>'
            ),
            "</svg>",
        ]
    )
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    if output.stat().st_size < 500:
        raise RuntimeError(f"Architecture SVG export is unexpectedly empty: {output}")
    return output
