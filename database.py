"""
Database management for TTS Benchmarking Tool
Handles persistent storage of results, ELO ratings, and historical data

Backends
--------
Two interchangeable backends are supported transparently:

* **SQLite** (default) — a local ``benchmark_data.db`` file. Used for local
  development and the legacy app. No configuration required.
* **PostgreSQL** (e.g. Neon) — enabled automatically when a ``DATABASE_URL``
  (``postgres://...`` / ``postgresql://...``) is provided via environment
  variable or Streamlit secrets. Use this for shared cloud deployments
  (Streamlit Community Cloud) so every vote persists in one place.

All existing methods keep using ``?`` placeholders and ``conn = self._connect()``;
a thin shim rewrites placeholders/DDL for Postgres so call sites stay identical.
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import pandas as pd


def _resolve_database_url() -> str:
    """Return a Postgres connection string if configured, else ''.

    Looks at the ``DATABASE_URL`` environment variable first (Streamlit Cloud
    exports top-level secrets to the environment), then falls back to
    ``st.secrets`` when Streamlit is importable.
    """
    url = (os.environ.get("DATABASE_URL") or "").strip()
    if url:
        return url
    try:  # optional: only present when running under Streamlit
        import streamlit as st  # noqa: WPS433 (local import by design)
        return str(st.secrets.get("DATABASE_URL", "")).strip()
    except Exception:
        return ""


class _CursorShim:
    """Wraps a DB-API cursor, translating SQLite syntax to Postgres on the fly.

    Only matters when ``use_postgres`` is True; for SQLite it is a passthrough.
    """

    def __init__(self, cursor, use_postgres: bool):
        self._cur = cursor
        self._pg = use_postgres

    @staticmethod
    def _translate(sql: str) -> str:
        sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
        sql = sql.replace(" DATETIME", " TIMESTAMP")
        # Make the legacy column-backfill ALTERs idempotent on Postgres.
        sql = sql.replace("ADD COLUMN ", "ADD COLUMN IF NOT EXISTS ")
        # psycopg uses %s placeholders; our queries use ?.
        sql = sql.replace("?", "%s")
        return sql

    def execute(self, sql: str, params=None):
        if self._pg:
            sql = self._translate(sql)
        if params is None:
            self._cur.execute(sql)
        else:
            self._cur.execute(sql, params)
        return self

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def __getattr__(self, name):
        return getattr(self._cur, name)


class _ConnShim:
    """Wraps a connection so ``cursor()`` yields a placeholder-translating cursor.

    Also remembers whether callers want dict-style rows (mirrors the previous
    ``conn.row_factory = sqlite3.Row`` usage).
    """

    def __init__(self, raw, use_postgres: bool, dict_rows: bool):
        self._raw = raw
        self._pg = use_postgres
        self._dict = dict_rows
        if not use_postgres and dict_rows:
            self._raw.row_factory = sqlite3.Row

    @property
    def raw(self):
        return self._raw

    def cursor(self):
        if self._pg:
            if self._dict:
                from psycopg.rows import dict_row
                cur = self._raw.cursor(row_factory=dict_row)
            else:
                cur = self._raw.cursor()
        else:
            cur = self._raw.cursor()
        return _CursorShim(cur, self._pg)

    def commit(self):
        self._raw.commit()

    def close(self):
        self._raw.close()


class BenchmarkDatabase:
    """Database manager for benchmark results and ELO ratings"""

    @staticmethod
    def default_db_path() -> str:
        """SQLite path; override with DATABASE_PATH for cloud deployments."""
        return os.environ.get("DATABASE_PATH", "benchmark_data.db")

    def __init__(self, db_path: Optional[str] = None, database_url: Optional[str] = None):
        self.database_url = (database_url or _resolve_database_url()).strip()
        self.use_postgres = self.database_url.startswith("postgres")
        self.db_path = db_path or self.default_db_path()
        if not self.use_postgres:
            parent = os.path.dirname(os.path.abspath(self.db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self.init_database()

    def _connect(self, dict_rows: bool = False) -> _ConnShim:
        """Open a connection to the active backend (SQLite or Postgres)."""
        if self.use_postgres:
            import psycopg
            raw = psycopg.connect(self.database_url)
        else:
            raw = sqlite3.connect(self.db_path)
        return _ConnShim(raw, self.use_postgres, dict_rows)
    
    def init_database(self):
        """Initialize database tables"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT,
                provider TEXT,
                voice TEXT,
                text TEXT,
                success BOOLEAN,
                latency_ms REAL,
                file_size_bytes INTEGER,
                error_message TEXT,
                metadata TEXT,
                timestamp DATETIME,
                category TEXT,
                word_count INTEGER,
                location_country TEXT,
                location_city TEXT,
                location_region TEXT
            )
        ''')
        
        try:
            cursor.execute('ALTER TABLE benchmark_results ADD COLUMN location_country TEXT')
        except:
            pass
        try:
            cursor.execute('ALTER TABLE benchmark_results ADD COLUMN location_city TEXT')
        except:
            pass
        try:
            cursor.execute('ALTER TABLE benchmark_results ADD COLUMN location_region TEXT')
        except:
            pass
        
        try:
            cursor.execute('ALTER TABLE benchmark_results ADD COLUMN latency_1 REAL DEFAULT 0')
        except:
            pass
        
        try:
            cursor.execute('ALTER TABLE benchmark_results ADD COLUMN ttfb REAL DEFAULT 0')
        except:
            pass
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS elo_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT UNIQUE,
                rating REAL DEFAULT 1000,
                games_played INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                last_updated DATETIME
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS provider_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT,
                total_tests INTEGER DEFAULT 0,
                successful_tests INTEGER DEFAULT 0,
                avg_latency REAL DEFAULT 0,
                avg_file_size REAL DEFAULT 0,
                last_updated DATETIME
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                test_type TEXT,
                providers TEXT,
                total_tests INTEGER,
                timestamp DATETIME,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                winner TEXT,
                loser TEXT,
                vote_type TEXT,
                text_sample TEXT,
                session_id TEXT,
                timestamp DATETIME,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS locale_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                locale TEXT,
                summary TEXT,
                comment_count INTEGER,
                model_used TEXT,
                created_at DATETIME,
                updated_at DATETIME,
                UNIQUE(locale)
            )
        ''')

        # ------------------------------------------------------------------
        # Voice Arena schema (auditable + reproducible).
        # A leaderboard cell traces back: ratings_run -> votes -> battles ->
        # the exact clips (hashes) + normalization params + config that made it.
        # ------------------------------------------------------------------
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS battles (
                battle_id TEXT PRIMARY KEY,
                language TEXT,
                item_id TEXT,
                item_text TEXT,
                strategy TEXT,
                anchor TEXT,
                competitor TEXT,
                is_anchor_pair INTEGER,
                provider_a TEXT,
                provider_b TEXT,
                left_provider TEXT,
                left_voice TEXT,
                right_provider TEXT,
                right_voice TEXT,
                position_seed INTEGER,
                left_clip_sha256 TEXT,
                right_clip_sha256 TEXT,
                normalization_params TEXT,
                session_id TEXT,
                location_country TEXT,
                location_city TEXT,
                location_region TEXT,
                created_at DATETIME
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                battle_id TEXT,
                outcome TEXT,                 -- 'A' (left), 'B' (right), or 'tie'
                winner_provider TEXT,         -- NULL for ties
                loser_provider TEXT,          -- NULL for ties
                left_provider TEXT,
                right_provider TEXT,
                language TEXT,
                comment TEXT,
                comment_deanonymized TEXT,
                rater_session TEXT,
                location_country TEXT,
                location_city TEXT,
                location_region TEXT,
                created_at DATETIME,
                FOREIGN KEY(battle_id) REFERENCES battles(battle_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ratings_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                scope TEXT,                   -- 'language' or 'overall'
                language TEXT,                -- NULL for overall
                inputs_hash TEXT,            -- hash of the vote set used
                code_version TEXT,
                engine_params TEXT,          -- JSON: solver/bootstrap/seed/weights
                results TEXT,                 -- JSON: per-provider strengths + CIs + elo
                n_battles INTEGER,
                n_votes INTEGER,
                created_at DATETIME
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_votes_battle ON votes(battle_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_votes_language ON votes(language)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_battles_language ON battles(language)')

        conn.commit()
        conn.close()
    
    def save_benchmark_result(self, result, test_id: str = None):
        """Save a benchmark result to database"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO benchmark_results 
            (test_id, provider, voice, text, success, latency_ms, file_size_bytes, 
             error_message, metadata, timestamp, category, word_count, 
             location_country, location_city, location_region, latency_1, ttfb)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            test_id or f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            result.provider,
            result.voice,
            result.sample.text if hasattr(result, 'sample') else "",
            result.success,
            result.latency_ms,
            result.file_size_bytes,
            result.error_message,
            json.dumps(result.metadata) if result.metadata else "{}",
            datetime.now(),
            getattr(result.sample, 'category', 'unknown') if hasattr(result, 'sample') else 'unknown',
            getattr(result.sample, 'word_count', 0) if hasattr(result, 'sample') else 0,
            getattr(result, 'location_country', 'Unknown'),
            getattr(result, 'location_city', 'Unknown'),
            getattr(result, 'location_region', 'Unknown'),
            getattr(result, 'latency_1', 0.0),
            getattr(result, 'ttfb', 0.0)
        ))
        
        conn.commit()
        conn.close()
        
        self.update_provider_stats(result.provider, result)
    
    def update_provider_stats(self, provider: str, result):
        """Update provider statistics"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM provider_stats WHERE provider = ?', (provider,))
        stats = cursor.fetchone()
        
        if stats:
            total_tests = stats[2] + 1
            successful_tests = stats[3] + (1 if result.success else 0)
            
            if result.success:
                old_avg_latency = stats[4]
                old_avg_file_size = stats[5]
                
                new_avg_latency = ((old_avg_latency * successful_tests) + result.latency_ms) / (successful_tests + 1) if successful_tests > 0 else result.latency_ms
                new_avg_file_size = ((old_avg_file_size * successful_tests) + result.file_size_bytes) / (successful_tests + 1) if successful_tests > 0 else result.file_size_bytes
            else:
                new_avg_latency = stats[4]
                new_avg_file_size = stats[5]
            
            cursor.execute('''
                UPDATE provider_stats 
                SET total_tests = ?, successful_tests = ?, avg_latency = ?, 
                    avg_file_size = ?, last_updated = ?
                WHERE provider = ?
            ''', (total_tests, successful_tests, new_avg_latency, new_avg_file_size, datetime.now(), provider))
        else:
            cursor.execute('''
                INSERT INTO provider_stats 
                (provider, total_tests, successful_tests, avg_latency, avg_file_size, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                provider, 1, 1 if result.success else 0,
                result.latency_ms if result.success else 0,
                result.file_size_bytes if result.success else 0,
                datetime.now()
            ))
        
        conn.commit()
        conn.close()
    
    def get_elo_rating(self, provider: str) -> float:
        """Get ELO rating for a provider"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('SELECT rating FROM elo_ratings WHERE provider = ?', (provider,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return result[0]
        else:
            self.init_elo_rating(provider)
            return 1000.0
    
    def init_elo_rating(self, provider: str, rating: float = 1000.0):
        """Initialize ELO rating for a new provider"""
        self.init_database()
        conn = self._connect()
        cursor = conn.cursor()
        
        if self.use_postgres:
            elo_sql = '''
                INSERT INTO elo_ratings
                (provider, rating, games_played, wins, losses, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (provider) DO NOTHING
            '''
        else:
            elo_sql = '''
                INSERT OR IGNORE INTO elo_ratings 
                (provider, rating, games_played, wins, losses, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            '''
        cursor.execute(elo_sql, (provider, rating, 0, 0, 0, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def update_elo_ratings(self, winner: str, loser: str, k_factor: int = 32):
        """Update ELO ratings using standard ELO formula
        
        IMPORTANT: This method should ONLY be called from blind test votes (user preferences).
        It should NOT be called from quick test results or benchmark comparisons based on
        technical metrics like latency or TTFB. ELO is based purely on user quality preferences.
        
        Standard ELO formula:
        - Expected score for winner: E_winner = 1 / (1 + 10^((loser_rating - winner_rating)/400))
        - Expected score for loser: E_loser = 1 / (1 + 10^((winner_rating - loser_rating)/400))
        - New rating = Old rating + K * (Actual score - Expected score)
        - Winner actual score = 1, Loser actual score = 0
        """
        import math
        
        # Ensure both providers are initialized
        self.init_elo_rating(winner)
        self.init_elo_rating(loser)
        
        winner_rating = self.get_elo_rating(winner)
        loser_rating = self.get_elo_rating(loser)
        
        # Calculate expected scores using EXACT standard ELO formula
        # E_X = 1 / (1 + 10^((R_Y - R_X) / 400))
        expected_winner = 1 / (1 + math.pow(10, (loser_rating - winner_rating) / 400))
        expected_loser = 1 / (1 + math.pow(10, (winner_rating - loser_rating) / 400))
        
        # Update ratings using EXACT formula: R'_X = R_X + K(S_X - E_X)
        # Winner: S_X = 1 (won), Loser: S_X = 0 (lost)
        new_winner_rating = winner_rating + k_factor * (1 - expected_winner)
        new_loser_rating = loser_rating + k_factor * (0 - expected_loser)
        
        # Ensure ratings don't go below 0
        if new_loser_rating < 0:
            new_loser_rating = 0
        
        conn = self._connect()
        cursor = conn.cursor()
        
        # Update winner: increase rating, increment wins
        cursor.execute('''
            UPDATE elo_ratings 
            SET rating = ?, games_played = games_played + 1, wins = wins + 1, last_updated = ?
            WHERE provider = ?
        ''', (new_winner_rating, datetime.now(), winner))
        
        # Update loser: decrease rating, increment losses
        cursor.execute('''
            UPDATE elo_ratings 
            SET rating = ?, games_played = games_played + 1, losses = losses + 1, last_updated = ?
            WHERE provider = ?
        ''', (new_loser_rating, datetime.now(), loser))
        
        conn.commit()
        conn.close()
        
        return new_winner_rating, new_loser_rating
    
    def get_all_elo_ratings(self) -> Dict[str, Dict]:
        """Get all ELO ratings"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM elo_ratings ORDER BY rating DESC')
        results = cursor.fetchall()
        
        conn.close()
        
        ratings = {}
        for row in results:
            ratings[row[1]] = {
                'rating': row[2],
                'games_played': row[3],
                'wins': row[4],
                'losses': row[5],
                'win_rate': (row[4] / row[3] * 100) if row[3] > 0 else 0,
                'last_updated': row[6]
            }
        
        return ratings
    
    def get_provider_stats(self) -> Dict[str, Dict]:
        """Get all provider statistics"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM provider_stats')
        results = cursor.fetchall()
        
        conn.close()
        
        stats = {}
        for row in results:
            stats[row[1]] = {
                'total_tests': row[2],
                'successful_tests': row[3],
                'success_rate': (row[3] / row[2] * 100) if row[2] > 0 else 0,
                'avg_latency': row[4],
                'avg_file_size': row[5],
                'last_updated': row[6]
            }
        
        return stats
    
    def get_recent_results(self, limit: int = 100) -> pd.DataFrame:
        """Get recent benchmark results as DataFrame"""
        conn = self._connect()
        
        query = '''
            SELECT * FROM benchmark_results 
            ORDER BY timestamp DESC 
            LIMIT ?
        '''
        if self.use_postgres:
            query = query.replace("?", "%s")
        df = pd.read_sql_query(query, conn.raw, params=(limit,))
        
        conn.close()
        return df
    
    def get_results_by_provider(self, provider: str, limit: int = 50) -> pd.DataFrame:
        """Get results for a specific provider"""
        conn = self._connect()
        
        query = '''
            SELECT * FROM benchmark_results 
            WHERE provider = ?
            ORDER BY timestamp DESC 
            LIMIT ?
        '''
        if self.use_postgres:
            query = query.replace("?", "%s")
        df = pd.read_sql_query(query, conn.raw, params=(provider, limit))
        
        conn.close()
        return df
    
    def clear_all_data(self):
        """Clear all data from database"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM benchmark_results')
        cursor.execute('DELETE FROM elo_ratings')
        cursor.execute('DELETE FROM provider_stats')
        cursor.execute('DELETE FROM test_sessions')
        cursor.execute('DELETE FROM user_votes')
        
        conn.commit()
        conn.close()
    
    def clear_old_data(self, days_old: int = 30):
        """Clear data older than specified days"""
        conn = self._connect()
        cursor = conn.cursor()
        
        days_old = int(days_old)
        if self.use_postgres:
            cursor.execute(
                "DELETE FROM benchmark_results WHERE timestamp < NOW() - (? || ' days')::interval",
                (days_old,),
            )
        else:
            cursor.execute(
                "DELETE FROM benchmark_results "
                "WHERE timestamp < datetime('now', '-{} days')".format(days_old)
            )
        
        conn.commit()
        conn.close()
    
    def export_data(self, format: str = 'json') -> str:
        """Export all data to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format.lower() == 'json':
            filename = f"benchmark_export_{timestamp}.json"
            
            data = {
                'elo_ratings': self.get_all_elo_ratings(),
                'provider_stats': self.get_provider_stats(),
                'recent_results': self.get_recent_results(1000).to_dict('records'),
                'export_timestamp': timestamp
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        elif format.lower() == 'csv':
            filename = f"benchmark_export_{timestamp}.csv"
            df = self.get_recent_results(1000)
            df.to_csv(filename, index=False)
        
        return filename
    
    def save_user_vote(self, winner: str, loser: str, text_sample: str, session_id: str = "default", vote_type: str = "user_preference", metadata: dict = None):
        """Save a user preference vote"""
        conn = self._connect()
        cursor = conn.cursor()
        
        # Determine vote source from session_id
        vote_source = 'quick_test'
        if 'blind_battle_2' in session_id:
            vote_source = 'ranked_blind_test'
        elif 'blind_battle' in session_id or 'blind_test' in session_id:
            vote_source = 'blind_test'
        
        vote_metadata = metadata or {}
        vote_metadata['vote_source'] = vote_source
        
        cursor.execute('''
            INSERT INTO user_votes 
            (winner, loser, vote_type, text_sample, session_id, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            winner, loser, vote_type, text_sample, session_id,
            datetime.now(), json.dumps(vote_metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def get_vote_statistics(self) -> Dict[str, Any]:
        """Get voting statistics"""
        conn = self._connect()
        cursor = conn.cursor()
        
        # Get total votes per provider
        cursor.execute('''
            SELECT winner, COUNT(*) as wins FROM user_votes GROUP BY winner
        ''')
        wins = dict(cursor.fetchall())
        
        cursor.execute('''
            SELECT loser, COUNT(*) as losses FROM user_votes GROUP BY loser  
        ''')
        losses = dict(cursor.fetchall())
        
        # Get recent votes
        cursor.execute('''
            SELECT winner, loser, timestamp FROM user_votes 
            ORDER BY timestamp DESC LIMIT 10
        ''')
        recent_votes = cursor.fetchall()
        
        conn.close()
        
        return {
            'wins': wins,
            'losses': losses,
            'recent_votes': recent_votes,
            'total_votes': sum(wins.values())
        }
    
    def get_ranked_blind_test_votes(self) -> List[tuple]:
        """Get all votes from Ranked Blind Test (blind_battle_2 sessions)"""
        conn = self._connect()
        cursor = conn.cursor()
        
        # Get votes where session_id contains 'blind_battle_2' or metadata contains 'ranked_blind_test'
        cursor.execute('''
            SELECT winner, loser, timestamp, metadata FROM user_votes 
            WHERE session_id LIKE '%blind_battle_2%' 
               OR metadata LIKE '%ranked_blind_test%'
            ORDER BY timestamp ASC
        ''')
        votes = cursor.fetchall()
        
        conn.close()
        return votes
    
    def get_fvs_votes(self) -> List[tuple]:
        """Get all votes from Falcon vs Zeroshot tests"""
        conn = self._connect()
        cursor = conn.cursor()
        
        # Get votes where session_id contains 'fvs' or metadata contains 'falcon_vs_zeroshot'
        cursor.execute('''
            SELECT winner, loser, text_sample, timestamp, metadata FROM user_votes 
            WHERE session_id LIKE '%fvs%' 
               OR metadata LIKE '%falcon_vs_zeroshot%'
            ORDER BY timestamp ASC
        ''')
        votes = cursor.fetchall()
        
        conn.close()
        return votes
    
    def save_locale_summary(self, locale: str, summary: str, comment_count: int, model_used: str = "gpt-4o"):
        """Save or update a summary for a locale"""
        conn = self._connect()
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        # Check if summary exists for this locale
        cursor.execute('SELECT id FROM locale_summaries WHERE locale = ?', (locale,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing summary
            cursor.execute('''
                UPDATE locale_summaries 
                SET summary = ?, comment_count = ?, model_used = ?, updated_at = ?
                WHERE locale = ?
            ''', (summary, comment_count, model_used, now, locale))
        else:
            # Insert new summary
            cursor.execute('''
                INSERT INTO locale_summaries (locale, summary, comment_count, model_used, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (locale, summary, comment_count, model_used, now, now))
        
        conn.commit()
        conn.close()
    
    def get_locale_summary(self, locale: str) -> Optional[Dict[str, Any]]:
        """Get the stored summary for a locale"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT locale, summary, comment_count, model_used, created_at, updated_at
            FROM locale_summaries
            WHERE locale = ?
        ''', (locale,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "locale": result[0],
                "summary": result[1],
                "comment_count": result[2],
                "model_used": result[3],
                "created_at": result[4],
                "updated_at": result[5]
            }
        return None
    
    def delete_locale_summary(self, locale: str):
        """Delete a stored summary for a locale (to force regeneration)"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM locale_summaries WHERE locale = ?', (locale,))
        conn.commit()
        conn.close()
    
    def get_latency_stats_by_provider(self) -> Dict[str, Dict]:
        """Get latency statistics including P95 for each provider"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT provider, latency_ms 
            FROM benchmark_results 
            WHERE success = 1 AND latency_ms > 0
            ORDER BY provider, latency_ms
        ''')
        results = cursor.fetchall()
        conn.close()
        
        provider_latencies = {}
        for provider, latency in results:
            if provider not in provider_latencies:
                provider_latencies[provider] = []
            provider_latencies[provider].append(latency)
        
        stats = {}
        for provider, latencies in provider_latencies.items():
            if not latencies:
                continue
            
            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)
            
            def percentile(data, p):
                if not data:
                    return 0
                index = (p / 100) * (len(data) - 1)
                if index.is_integer():
                    return data[int(index)]
                else:
                    lower = data[int(index)]
                    upper = data[int(index) + 1]
                    return lower + (upper - lower) * (index - int(index))
            
            stats[provider] = {
                'avg_latency': sum(latencies) / n if n > 0 else 0,
                'median_latency': percentile(latencies_sorted, 50),
                'p90_latency': percentile(latencies_sorted, 90),
                'p95_latency': percentile(latencies_sorted, 95),
                'p99_latency': percentile(latencies_sorted, 99),
                'min_latency': latencies_sorted[0] if latencies_sorted else 0,
                'max_latency': latencies_sorted[-1] if latencies_sorted else 0,
                'total_tests': n
            }
        
        return stats
    
    def get_ping_stats_by_provider(self) -> Dict[str, Dict]:
        """Get network latency (latency_1) statistics for each provider"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT provider, latency_1 
            FROM benchmark_results 
            WHERE success = 1 AND latency_1 > 0
            ORDER BY provider, latency_1
        ''')
        results = cursor.fetchall()
        conn.close()
        
        provider_pings = {}
        for provider, ping in results:
            if provider not in provider_pings:
                provider_pings[provider] = []
            provider_pings[provider].append(ping)
        
        stats = {}
        for provider, pings in provider_pings.items():
            if not pings:
                continue
            
            pings_sorted = sorted(pings)
            n = len(pings_sorted)
            
            def percentile(data, p):
                if not data:
                    return 0
                index = (p / 100) * (len(data) - 1)
                if index.is_integer():
                    return data[int(index)]
                else:
                    lower = data[int(index)]
                    upper = data[int(index) + 1]
                    return lower + (upper - lower) * (index - int(index))
            
            stats[provider] = {
                'avg_ping': sum(pings) / n if n > 0 else 0,
                'median_ping': percentile(pings_sorted, 50),
                'p90_ping': percentile(pings_sorted, 90),
                'p95_ping': percentile(pings_sorted, 95),
                'p99_ping': percentile(pings_sorted, 99),
                'min_ping': pings_sorted[0] if pings_sorted else 0,
                'max_ping': pings_sorted[-1] if pings_sorted else 0,
                'total_tests': n
            }
        
        return stats
    
    def get_ttfb_stats_by_provider(self) -> Dict[str, Dict]:
        """Get TTFB (Time to First Byte) statistics for each provider"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT provider, ttfb 
            FROM benchmark_results 
            WHERE success = 1 AND ttfb > 0
            ORDER BY provider, ttfb
        ''')
        results = cursor.fetchall()
        conn.close()
        
        provider_ttfbs = {}
        for provider, ttfb in results:
            if provider not in provider_ttfbs:
                provider_ttfbs[provider] = []
            provider_ttfbs[provider].append(ttfb)
        
        stats = {}
        for provider, ttfbs in provider_ttfbs.items():
            if not ttfbs:
                continue
            
            ttfbs_sorted = sorted(ttfbs)
            n = len(ttfbs_sorted)
            
            def percentile(data, p):
                if not data:
                    return 0
                index = (p / 100) * (len(data) - 1)
                if index.is_integer():
                    return data[int(index)]
                else:
                    lower = data[int(index)]
                    upper = data[int(index) + 1]
                    return lower + (upper - lower) * (index - int(index))
            
            stats[provider] = {
                'avg_ttfb': sum(ttfbs) / n if n > 0 else 0,
                'median_ttfb': percentile(ttfbs_sorted, 50),
                'p90_ttfb': percentile(ttfbs_sorted, 90),
                'p95_ttfb': percentile(ttfbs_sorted, 95),
                'p99_ttfb': percentile(ttfbs_sorted, 99),
                'min_ttfb': ttfbs_sorted[0] if ttfbs_sorted else 0,
                'max_ttfb': ttfbs_sorted[-1] if ttfbs_sorted else 0,
                'total_tests': n
            }
        
        return stats

    # ======================================================================
    # Voice Arena: battles / votes / ratings_runs
    # ======================================================================
    def create_battle(self, plan, left_clip_sha256: str, right_clip_sha256: str,
                      normalization_params: dict = None, session_id: str = "default",
                      location: dict = None):
        """Persist a scheduled battle (its clips' hashes + normalization params).

        `plan` is a scheduler.BattlePlan (or any object with the same fields).
        """
        location = location or {}
        conn = self._connect()
        cursor = conn.cursor()
        cols = '''
            (battle_id, language, item_id, item_text, strategy, anchor, competitor,
             is_anchor_pair, provider_a, provider_b, left_provider, left_voice,
             right_provider, right_voice, position_seed, left_clip_sha256,
             right_clip_sha256, normalization_params, session_id,
             location_country, location_city, location_region, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''
        if self.use_postgres:
            insert_sql = "INSERT INTO battles" + cols + " ON CONFLICT (battle_id) DO NOTHING"
        else:
            insert_sql = "INSERT OR REPLACE INTO battles" + cols
        cursor.execute(insert_sql, (
            plan.battle_id, plan.language, plan.item_id, plan.item_text, plan.strategy,
            plan.anchor, plan.competitor, 1 if plan.is_anchor_pair else 0,
            plan.provider_a, plan.provider_b, plan.left_provider, plan.left_voice,
            plan.right_provider, plan.right_voice, plan.position_seed,
            left_clip_sha256, right_clip_sha256,
            json.dumps(normalization_params or {}), session_id,
            location.get("country", "Unknown"), location.get("city", "Unknown"),
            location.get("region", "Unknown"), datetime.now(),
        ))
        conn.commit()
        conn.close()

    def get_battle(self, battle_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect(dict_rows=True)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM battles WHERE battle_id = ?', (battle_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def record_vote(self, battle_id: str, outcome: str, comment: str = "",
                    comment_deanonymized: str = "", rater_session: str = "default",
                    location: dict = None) -> bool:
        """Record a Left/Right/Tie vote, deriving winner/loser from the battle.

        outcome: 'A' (left better), 'B' (right better), or 'tie'.
        """
        outcome = (outcome or "").strip().lower()
        norm = {"a": "A", "left": "A", "b": "B", "right": "B",
                "tie": "tie", "same": "tie"}.get(outcome, None)
        if norm is None:
            raise ValueError(f"Invalid outcome '{outcome}' (expected A/B/tie).")

        battle = self.get_battle(battle_id)
        if not battle:
            raise ValueError(f"Unknown battle_id '{battle_id}'.")

        left_p = battle["left_provider"]
        right_p = battle["right_provider"]
        if norm == "A":
            winner, loser = left_p, right_p
        elif norm == "B":
            winner, loser = right_p, left_p
        else:
            winner, loser = None, None

        location = location or {}
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO votes
            (battle_id, outcome, winner_provider, loser_provider, left_provider,
             right_provider, language, comment, comment_deanonymized, rater_session,
             location_country, location_city, location_region, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            battle_id, norm, winner, loser, left_p, right_p, battle["language"],
            comment or "", comment_deanonymized or "", rater_session,
            location.get("country", "Unknown"), location.get("city", "Unknown"),
            location.get("region", "Unknown"), datetime.now(),
        ))
        conn.commit()
        conn.close()
        return True

    def get_outcomes(self, language: str = None) -> List[Dict[str, Any]]:
        """Outcomes for the rating engine: (provider_a, provider_b, outcome).

        provider_a = left provider, provider_b = right provider, outcome in
        {'A','B','tie'} where 'A' means provider_a was preferred. Joins votes to
        battles so the labeling matches the served clips.
        """
        conn = self._connect()
        cursor = conn.cursor()
        if language:
            cursor.execute('''
                SELECT left_provider, right_provider, outcome
                FROM votes WHERE language = ?
            ''', (language,))
        else:
            cursor.execute('SELECT left_provider, right_provider, outcome FROM votes')
        rows = cursor.fetchall()
        conn.close()
        return [
            {"provider_a": a, "provider_b": b, "outcome": o}
            for (a, b, o) in rows if a and b
        ]

    def get_voice_outcomes(self, language: str = None) -> List[Dict[str, Any]]:
        """Per-vote rows joined to battle voices (for per-voice analysis).

        Returns left/right provider + the exact voice id served on each side,
        plus the outcome ('A'=left preferred, 'B'=right preferred, 'tie').
        """
        conn = self._connect()
        cursor = conn.cursor()
        base = '''
            SELECT b.left_provider, b.right_provider, b.left_voice, b.right_voice, v.outcome
            FROM votes v JOIN battles b ON v.battle_id = b.battle_id
        '''
        if language:
            cursor.execute(base + ' WHERE v.language = ?', (language,))
        else:
            cursor.execute(base)
        rows = cursor.fetchall()
        conn.close()
        return [
            {"left_provider": lp, "right_provider": rp,
             "left_voice": lv, "right_voice": rv, "outcome": o}
            for (lp, rp, lv, rv, o) in rows if lp and rp
        ]

    def get_head_to_head(self, language: str = None) -> Dict[str, Dict[str, int]]:
        """Per-pair tallies keyed by 'p1|p2' (sorted) -> wins/ties counts.

        Returns { 'pA|pB': {'a_wins':.., 'b_wins':.., 'ties':.., 'n':..} } where
        a/b follow the sorted provider order for stability.
        """
        outcomes = self.get_outcomes(language)
        grid: Dict[str, Dict[str, int]] = {}
        for o in outcomes:
            pa, pb, res = o["provider_a"], o["provider_b"], o["outcome"]
            lo, hi = sorted([pa, pb])
            key = f"{lo}|{hi}"
            cell = grid.setdefault(key, {"a_wins": 0, "b_wins": 0, "ties": 0, "n": 0})
            cell["n"] += 1
            if res == "tie":
                cell["ties"] += 1
            else:
                winner = pa if res == "A" else pb
                if winner == lo:
                    cell["a_wins"] += 1
                else:
                    cell["b_wins"] += 1
        return grid

    def get_languages_with_votes(self) -> List[str]:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT language FROM votes WHERE language IS NOT NULL')
        langs = [r[0] for r in cursor.fetchall()]
        conn.close()
        return langs

    def get_vote_counts(self) -> Dict[str, int]:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM votes')
        total = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM battles')
        battles = cursor.fetchone()[0]
        conn.close()
        return {"votes": total, "battles": battles}

    def get_arena_comments(self, language: str = None) -> List[Dict[str, Any]]:
        """De-anonymized comments (for the per-language OpenAI summary)."""
        conn = self._connect(dict_rows=True)
        cursor = conn.cursor()
        if language:
            cursor.execute('''SELECT * FROM votes
                WHERE language = ? AND comment_deanonymized != '' ''', (language,))
        else:
            cursor.execute('''SELECT * FROM votes WHERE comment_deanonymized != '' ''')
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def save_ratings_run(self, run_id: str, scope: str, language: str,
                         inputs_hash: str, code_version: str, engine_params: dict,
                         results: dict, n_battles: int, n_votes: int):
        """Persist a versioned snapshot of a fit (reproducible published number)."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO ratings_runs
            (run_id, scope, language, inputs_hash, code_version, engine_params,
             results, n_battles, n_votes, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        ''', (
            run_id, scope, language, inputs_hash, code_version,
            json.dumps(engine_params, default=str), json.dumps(results, default=str),
            n_battles, n_votes, datetime.now(),
        ))
        conn.commit()
        conn.close()

    def get_latest_ratings_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self._connect(dict_rows=True)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM ratings_runs ORDER BY created_at DESC LIMIT ?', (limit,))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows


db = BenchmarkDatabase()