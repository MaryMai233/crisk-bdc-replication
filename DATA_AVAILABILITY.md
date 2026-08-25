# Data availability

CRSP and Compustat inputs are proprietary and intentionally excluded from this public repository. Users with WRDS access should obtain daily CRSP security data, Compustat North America quarterly data, and the CRSP/Compustat link table, then place them in the documented module `Data/Raw/` directories.

The cumulative-75-percent KOL robustness uses the official VanEck N-PORT schedule dated 30 September 2020. The selected 15 securities and market values are encoded in `01_Bank_CRISK_Replication/Code/08_build_kol_top75_continuation.py`, which also records the SEC source URL and downloads the associated public market and FX series.

SEC filing inputs and other public market-series inputs are described by the download and extraction programs in each module. The private replication archive supplied with the paper contains licensed raw and processed files for authorized use; those files must not be redistributed through this public repository.
