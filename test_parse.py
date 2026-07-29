import re
from datetime import datetime

filename = "20260707_095_Dep_Ran_Ado_14_7_115c_Colorado2025.czi"

def parse_date(fname):
    # Try to find the pattern: _[Day]_[Month]_
    # Wait, the species could be anything, so we look for _(\d{1,2})_(\d{1,2})_
    # Let's match from the end if possible, or just search for the first occurrence of this pattern after the species.
    # Actually, the format is [Date]_[Num]_Dep_[Species]_[Day]_[Month]_[ID]c_Colorado...
    match = re.search(r'Dep_[a-zA-Z]+_[a-zA-Z]+_(\d{1,2})_(\d{1,2})_', fname, re.IGNORECASE)
    if match:
        day = int(match.group(1))
        month = int(match.group(2))
    else:
        # Fallback to just finding any _\d+_\d+_
        match = re.search(r'_(\d{1,2})_(\d{1,2})_', fname)
        if match:
            day = int(match.group(1))
            month = int(match.group(2))
        else:
            return None, None, None

    year = 2025 # Should we assume 2025 from the Colorado2025 string?
    if "2025" in fname:
        year = 2025
    elif "2026" in fname:
        year = 2026

    date_obj = datetime(year, month, day)
    day_of_year = date_obj.timetuple().tm_yday
    return day, month, day_of_year

print(parse_date(filename))
