"""
SQL 安全校验模块
"""
import re

FORBIDDEN_PATTERNS = [
    re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b", re.IGNORECASE),
    re.compile(r"\bpatients\b[^;]*\b(phone|id_card)\b", re.IGNORECASE),
]


def validate_sql(sql: str) -> tuple[bool, str]:
    """
    校验 SQL 安全性。
    返回 (is_valid, validated_sql_or_error_message)
    """
    stripped = sql.strip().rstrip(";")

    if not stripped.upper().startswith("SELECT"):
        return False, "只允许 SELECT 查询"

    for pattern in FORBIDDEN_PATTERNS:
        if pattern.search(stripped):
            return False, "查询包含禁止的操作或字段"

    if "LIMIT" not in stripped.upper():
        stripped += " LIMIT 100"

    return True, stripped


def apply_role_filter(sql: str, role: str, department_id: int | None = None) -> str:
    """
    根据角色注入数据过滤条件。
    doctor 角色只能看自己科室的数据。
    """
    if role == "doctor" and department_id:
        if "WHERE" in sql.upper():
            sql = sql.replace("WHERE", f"WHERE department_id = {department_id} AND", 1)
        elif "LIMIT" in sql.upper():
            sql = sql.replace("LIMIT", f"WHERE department_id = {department_id} LIMIT", 1)
    return sql
