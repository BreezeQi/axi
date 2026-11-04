"""字段文档与查看选择共用的 JSONPath 解析和名称转义。"""

import json
import re

from axi.output.schema import MAX_DATA_DEPTH

_SEGMENT = re.compile(
    r"\.([A-Za-z_\x80-\ud7ff\ue000-\U0010ffff][A-Za-z_0-9\x80-\ud7ff\ue000-\U0010ffff]*)"
    r"""|\[("(?:[^"\\\x00-\x1f]|\\.)*"|'(?:[^'\\\x00-\x1f]|\\.)*')\]"""
    r"|\[(\*|(?:0|[1-9][0-9]*):?)\]"
)


def name_segment(name: str) -> str:
    """生成属性路径片段，特殊名称用引号保留，避免与层级和数组位置混淆。"""
    return (
        f".{name}"
        if name.isascii() and name.isidentifier()
        else f"[{json.dumps(name, ensure_ascii=False)}]"
    )


def segment_name(segment: str) -> str | None:
    """从规范化路径片段还原属性名；数组位置返回 None。"""
    if segment.startswith("."):
        return segment[1:]
    return json.loads(segment[1:-1]) if segment.startswith('["') else None


def parse_path(path: str) -> tuple[str, ...]:
    """解析 Schema 声明位置的选择路径，不对实际数据执行查询。"""
    if not path.startswith("$"):
        raise ValueError("Field paths must start with $. Use a JSONPath array.")
    segments: list[str] = []
    offset = 1
    while offset < len(path):
        match = _SEGMENT.match(path, offset)
        if match is None:
            raise ValueError(
                f"Unsupported field JSONPath: {path!r}. Use a declared position."
            )
        name, quoted, position = match.groups()
        if quoted is not None:
            if quoted[0] == "'":
                # 单引号名称沿用 JSON 转义，但用 \' 转义单引号。
                body = re.sub(
                    r'\\.|"',
                    lambda token: {"\\'": "'", '"': '\\"'}.get(token[0], token[0]),
                    quoted[1:-1],
                )
                quoted = f'"{body}"'
            name = json.loads(quoted)
        segments.append(name_segment(name) if name is not None else f"[{position}]")
        offset = match.end()
    if len(segments) > MAX_DATA_DEPTH:
        raise ValueError(
            f"Field path exceeds {MAX_DATA_DEPTH} levels. Use a parent subtree."
        )
    return tuple(segments)
