# PHONY target to ensure "clean" is run as a command and not as a file check.
.PHONY: clean

# clean command definition
clean:
	@echo "Cleaning up..."
	rm -f *_new*               # Remove all files containing with '_new'
	rm -f *.data               # Remove all .data files
	rm -f *.xyz                # Remove all .xyz files
	rm -f *.log                # Remove all .log files
	rm -f *.lammps-data        # Remove all .lammps-data files
	rm -f *.lammps*            # Remove all .lammps files
	rm -rf minim_output/       # Remove the minim_output directory and its contents
	@echo "Cleaning completed."
