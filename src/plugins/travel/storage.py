from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import aiosqlite


@dataclass(frozen=True)
class CheckinResult:
    """一次打卡写入的结果。"""

    status: str
    country: str
    previous_country: str | None = None


@dataclass(frozen=True)
class AttendanceStats:
    """指定群、国家和日期范围内的出勤统计。"""

    country: str
    start_date: date
    end_date: date
    daily_counts: dict[date, int]
    unique_people: int

    @property
    def total_checkins(self) -> int:
        return sum(self.daily_counts.values())

    @property
    def active_days(self) -> int:
        return sum(count > 0 for count in self.daily_counts.values())

    @property
    def peak_count(self) -> int:
        return max(self.daily_counts.values(), default=0)

    @property
    def peak_date(self) -> date | None:
        if self.peak_count == 0:
            return None
        return min(
            day
            for day, count in self.daily_counts.items()
            if count == self.peak_count
        )


class TravelStorage:
    """用单表 SQLite 保存群内每日位置，各群通过 group_id 隔离。"""

    def __init__(self, path: str):
        self.path = Path(path)
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is not None:
            return self._conn
        async with self._lock:
            if self._conn is not None:
                return self._conn
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(self.path)
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    country TEXT NOT NULL,
                    checkin_date TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (group_id, user_id, checkin_date)
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_checkins_group_country_date
                ON checkins (group_id, country, checkin_date)
                """
            )
            await conn.commit()
            self._conn = conn
            return conn

    async def record_checkin(
        self,
        *,
        group_id: int,
        user_id: int,
        country: str,
        checkin_date: date,
        checked_at: datetime,
    ) -> CheckinResult:
        """
        记录当天位置。

        同一成员同一天只保留一个国家；再次提交相同国家保持幂等，
        提交其他国家则更新当天位置，避免地理分布重复计数。
        """

        conn = await self._get_conn()
        day_text = checkin_date.isoformat()
        time_text = checked_at.isoformat(timespec="seconds")
        async with self._lock:
            cursor = await conn.execute(
                """
                SELECT country
                FROM checkins
                WHERE group_id = ? AND user_id = ? AND checkin_date = ?
                """,
                (int(group_id), int(user_id), day_text),
            )
            row = await cursor.fetchone()
            await cursor.close()

            if row is None:
                await conn.execute(
                    """
                    INSERT INTO checkins (
                        group_id, user_id, country, checkin_date,
                        created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(group_id),
                        int(user_id),
                        country,
                        day_text,
                        time_text,
                        time_text,
                    ),
                )
                await conn.commit()
                return CheckinResult(status="created", country=country)

            previous_country = str(row[0])
            if previous_country == country:
                return CheckinResult(status="unchanged", country=country)

            await conn.execute(
                """
                UPDATE checkins
                SET country = ?, updated_at = ?
                WHERE group_id = ? AND user_id = ? AND checkin_date = ?
                """,
                (
                    country,
                    time_text,
                    int(group_id),
                    int(user_id),
                    day_text,
                ),
            )
            await conn.commit()
            return CheckinResult(
                status="updated",
                country=country,
                previous_country=previous_country,
            )

    async def query_attendance(
        self,
        *,
        group_id: int,
        country: str,
        start_date: date,
        end_date: date,
    ) -> AttendanceStats:
        if start_date > end_date:
            raise ValueError("start_date 不能晚于 end_date")

        conn = await self._get_conn()
        query_args = (
            int(group_id),
            country,
            start_date.isoformat(),
            end_date.isoformat(),
        )
        cursor = await conn.execute(
            """
            SELECT checkin_date, COUNT(*)
            FROM checkins
            WHERE group_id = ?
              AND country = ?
              AND checkin_date BETWEEN ? AND ?
            GROUP BY checkin_date
            ORDER BY checkin_date
            """,
            query_args,
        )
        rows = await cursor.fetchall()
        await cursor.close()

        cursor = await conn.execute(
            """
            SELECT COUNT(DISTINCT user_id)
            FROM checkins
            WHERE group_id = ?
              AND country = ?
              AND checkin_date BETWEEN ? AND ?
            """,
            query_args,
        )
        unique_people_row = await cursor.fetchone()
        await cursor.close()

        daily_counts: dict[date, int] = {}
        current = start_date
        while current <= end_date:
            daily_counts[current] = 0
            current = date.fromordinal(current.toordinal() + 1)
        for day_text, count in rows:
            daily_counts[date.fromisoformat(day_text)] = int(count)

        return AttendanceStats(
            country=country,
            start_date=start_date,
            end_date=end_date,
            daily_counts=daily_counts,
            unique_people=int(unique_people_row[0] or 0),
        )

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None
