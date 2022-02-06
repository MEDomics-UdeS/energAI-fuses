## Module Details: reports/parse_results_[...].py

### Description

The following scripts can be used to parse the results generated
when executing phases A, B and C experiments:

- parse_results_phase_A.py
- parse_results_phase_B.py
- parse_results_phase_C.py

The following script can be used to parse the results of all phases 
and create a table with the total execution time per experiment phase:

- parse_results_time.py

The results parsed are formatted in LaTeX tables with proper section 
titles and headers and saved into a text file which can be 
directly copied and pasted into a LaTeX compiler, such as Overleaf.

### Examples of basic use:

To parse results from phase A:
```
python reports/parse_results_phase_A.py
```