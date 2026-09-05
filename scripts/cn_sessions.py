"""Versioned domestic ETF session baseline; unknown years fail closed."""
import argparse
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

VERSION = 'CN_SESSIONS_2026_V1'
SOURCE = 'https://www.sse.com.cn/disclosure/announcement/general/c/c_20251222_10802507.shtml'
TZ = ZoneInfo('Asia/Shanghai')
HOLIDAYS = [('2026-01-01','2026-01-03'), ('2026-02-15','2026-02-23'),
            ('2026-04-04','2026-04-06'), ('2026-05-01','2026-05-05'),
            ('2026-06-19','2026-06-21'), ('2026-09-25','2026-09-27'),
            ('2026-10-01','2026-10-07')]

def session(d):
    d = date.fromisoformat(str(d)[:10])
    if d.year != 2026:
        raise ValueError('CALENDAR_YEAR_NOT_REGISTERED')
    return d.weekday() < 5 and not any(date.fromisoformat(a) <= d <= date.fromisoformat(b) for a,b in HOLIDAYS)

def previous_session(as_of):
    d = date.fromisoformat(str(as_of)[:10]) - timedelta(days=1)
    while not session(d): d -= timedelta(days=1)
    return d.isoformat()

def next_execution(signal_date, available_at):
    """Planned close strictly after signal and real availability; never a fill."""
    observed = datetime.fromisoformat(available_at)
    if observed.tzinfo is None: raise ValueError('AVAILABILITY_TIMEZONE_REQUIRED')
    d = date.fromisoformat(str(signal_date)[:10]) + timedelta(days=1)
    d = max(d, observed.astimezone(TZ).date())
    while not session(d) or datetime.combine(d,time(15),TZ) <= observed:
        d += timedelta(days=1)
    return d.isoformat()

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--as-of',required=True)
    print(previous_session(p.parse_args().as_of))
