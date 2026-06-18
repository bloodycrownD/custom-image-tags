"""train_group_map 加载与条件组索引解析。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

ConditionMode = Literal["train_group", "author", "none"]

UNKNOWN_IDX = 0


class GroupMap:
    """封装 train_group_map.json，支持 author / train_group / none 三种条件模式。"""

    def __init__(
        self,
        groups: dict[str, int],
        authors: dict[str, Any],
        images: dict[str, str] | None = None,
    ):
        """
        Args:
            groups: 组名 → Embedding 索引（index 0 保留为 unknown，不在此 dict 中）。
            authors: author_id → 组名（train_group 模式）或 Embedding 索引（author 模式）。
            images: 可选，相对图片路径 → 组名，用于按图精确映射 train_group。
        """
        self.groups = groups
        self.authors = authors
        self.images = images or {}

    @classmethod
    def load(cls, path: str | Path) -> GroupMap:
        """从 JSON 文件加载组映射表。"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            groups=data.get("groups", {}),
            authors=data.get("authors", {}),
            images=data.get("images"),
        )

    def save(self, path: str | Path) -> None:
        """将组映射表写入 JSON 文件。"""
        payload = {
            "groups": self.groups,
            "authors": self.authors,
        }
        if self.images:
            payload["images"] = self.images
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @property
    def num_groups(self) -> int:
        """Embedding 表大小（不含 unknown 槽位）。"""
        if not self.groups:
            max_author_idx = 0
            for value in self.authors.values():
                if isinstance(value, int):
                    max_author_idx = max(max_author_idx, value)
            return max_author_idx
        return max(self.groups.values(), default=0)

    def resolve(
        self,
        author: str,
        image_path: str | None = None,
        condition: ConditionMode = "train_group",
    ) -> int:
        """
        解析单张图的条件 Embedding 索引。

        index 0 始终表示 unknown。
        """
        if condition == "none":
            return UNKNOWN_IDX

        if condition == "author":
            value = self.authors.get(author, UNKNOWN_IDX)
            if isinstance(value, int):
                return value if value >= 0 else UNKNOWN_IDX
            return self.groups.get(value, UNKNOWN_IDX)

        # train_group 模式：优先按图片路径，再回退 author → 组名
        group_name: str | None = None
        if image_path and image_path in self.images:
            group_name = self.images[image_path]
        else:
            author_val = self.authors.get(author)
            if isinstance(author_val, str):
                group_name = author_val
            elif isinstance(author_val, int):
                return author_val if author_val > 0 else UNKNOWN_IDX

        if group_name is None:
            return UNKNOWN_IDX
        return self.groups.get(group_name, UNKNOWN_IDX)

    def to_dict(self) -> dict[str, Any]:
        """序列化为 checkpoint 可存储的字典。"""
        result: dict[str, Any] = {
            "groups": self.groups,
            "authors": self.authors,
        }
        if self.images:
            result["images"] = self.images
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroupMap:
        """从 checkpoint 字典还原。"""
        return cls(
            groups=data.get("groups", {}),
            authors=data.get("authors", {}),
            images=data.get("images"),
        )
