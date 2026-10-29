# make all  == bash run_all.sh      make smoke == full pipeline on fake data (2 minutes, meaningless numbers)
.PHONY: all smoke clean
all:
	bash run_all.sh
smoke:
	MEDISCAN_DATA=data_synth python tools/make_synthetic_data.py && MEDISCAN_DATA=data_synth QUICK=1 bash run_all.sh
clean:
	rm -rf cache results splits models deploy reports/fragments reports/figures reports/Part1_audit.md reports/Report.md reports/CEO_report.md
