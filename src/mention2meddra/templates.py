from __future__ import annotations

from typing import Any


TEMPLATE_NAMES = ("full", "pt_only", "pt_llt")


def clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def render_candidate_text(row: dict[str, Any], template_name: str = "full") -> str:
    if template_name not in TEMPLATE_NAMES:
        raise ValueError(f"unknown template_name={template_name!r}; expected one of {TEMPLATE_NAMES}")

    pt = str(row.get("candidate_pt_name", row.get("pt_name", ""))).strip()
    llts = clean_list(row.get("candidate_llt_names", row.get("llt_names")))
    hlt = str(row.get("candidate_hlt_name", row.get("hlt_name", ""))).strip()
    hlgt = str(row.get("candidate_hlgt_name", row.get("hlgt_name", ""))).strip()
    soc = str(row.get("candidate_soc_name", row.get("soc_name", ""))).strip()

    if template_name == "pt_only":
        return f"PT: {pt}"

    alias_str = " | ".join(llts[:8])
    if template_name == "pt_llt":
        return f"PT: {pt}；LLT别名: {alias_str}" if alias_str else f"PT: {pt}"

    parts = [f"PT: {pt}"]
    if alias_str:
        parts.append(f"LLT别名: {alias_str}")
    if hlt:
        parts.append(f"HLT: {hlt}")
    if hlgt:
        parts.append(f"HLGT: {hlgt}")
    if soc:
        parts.append(f"SOC: {soc}")
    return "；".join(parts)
