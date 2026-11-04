"""准备工具与字段语料，记录文本归属及字段名、JSONPath 精确索引。"""

from pydantic import BaseModel

from axi.models import ToolMeta
from axi.output.documents import FieldDocuments, OutputField
from axi.output.paths import parse_path

type ExactFields = dict[str, list[int]]


def find_exact_fields(index: ExactFields, query: str) -> list[int]:
    """按字段字面名称或规范化 JSONPath 查找，等价路径写法定位同一声明。"""
    matches = index.get(query.casefold(), [])
    if not query.startswith("$"):
        return matches
    try:
        path = "$" + "".join(parse_path(query))
    except ValueError:
        # search 接受任意文本；不支持的路径表达式继续作为普通查询。
        return matches
    return list(dict.fromkeys((*matches, *index.get(path.casefold(), []))))


class SearchDocument(BaseModel):
    """记录检索文本所属的工具及字段；根说明没有字段条目。"""

    tool: int
    field: OutputField | None = None


class SearchCorpus(BaseModel):
    """工具说明与字段分别建库，字段文本、归属和精确索引保持对应。"""

    tool_texts: list[str]
    output_texts: list[str]
    output_documents: list[SearchDocument]
    exact_fields: ExactFields


def build_corpus(
    tools: list[ToolMeta], field_documents: list[FieldDocuments]
) -> SearchCorpus:
    """按相同顺序准备文本与归属，建库后仅需保留归属和字段索引。"""
    tool_texts = []
    output_texts = []
    output_documents = []
    exact: ExactFields = {}
    for owner, (tool, documentation) in enumerate(
        zip(tools, field_documents, strict=True)
    ):
        tool_texts.append(f"{tool.full_name} {tool.description}")
        if documentation.root_text:
            # Schema 根说明参与检索，但不占用命中字段摘要名额。
            output_documents.append(SearchDocument.model_construct(tool=owner))
            output_texts.append(documentation.root_text)
        for field in documentation.fields:
            path = field.path
            document_id = len(output_documents)
            output_documents.append(
                SearchDocument.model_construct(
                    tool=owner,
                    field=field,
                )
            )
            output_texts.append(field.text)
            for name in dict.fromkeys((path.casefold(), field.name.casefold())):
                if name:
                    exact.setdefault(name, []).append(document_id)
    return SearchCorpus.model_construct(
        tool_texts=tool_texts,
        output_texts=output_texts,
        output_documents=output_documents,
        exact_fields=exact,
    )
