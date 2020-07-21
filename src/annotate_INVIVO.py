import csv
import pyopenms
import re
import pprint
from pyopenms import *
import pandas as pd
import matplotlib.pyplot as plt

# load manual validated results
for filename in {"AStuetzer_100118_120118_HeLaUV.csv"}: # csv filenames
  with open(filename) as csv_file:
    manual_results = list(csv.reader(csv_file, delimiter=';'))
manual_results.pop(0) # remove header
n_manual_validated = len(manual_results)
print("Input: " + str(n_manual_validated) + " manual validated hits.")

for filename in {"entrapment_slow_perc_1.0000_XLs.idXML", "entrapment_slow_autotune_perc_1.0000_XLs.idXML"}:
#for filename in {"entrapment_slow_autotune_1.0000_XLs.idXML"}:
#for filename in {"entrapment_0.9990_XLs.idXML"}:
  # load identification results
  prot_ids = []; pep_ids = []
#IdXMLFile().load("entrapment_perc_0.9990_XLs.idXML", prot_ids, pep_ids)
  IdXMLFile().load(filename, prot_ids, pep_ids)

  xl2minFDR = dict()
  PSM2FDR = dict()

  previous_scan_index = 0

  # Iterate over all protein hits
  for peptide_id in pep_ids:
    scan_index = peptide_id.getMetaValue("scan_index")
    if scan_index > n_manual_validated:
      break

    while previous_scan_index < scan_index: #output lines for unidentified PSMs
      manual_peptide = re.sub(r'\..*', '', manual_results[previous_scan_index][4])
      print("No ID " + str(previous_scan_index) + ": "+ str(manual_peptide))
      previous_scan_index+=1

    hit = peptide_id.getHits()[0]
    score = hit.getScore()
    PSM = hit.getSequence().toString().decode() + "_" + str(int(peptide_id.getRT()))
    #print(str(peptide_id.getRT())+";"+str(peptide_id.getMZ())+";"+hit.getSequence().toString().decode()+";"+hit.getMetaValue("NuXL:NA").decode()+";"+str(hit.getScore()))
    peptide = hit.getSequence().toUnmodifiedString().decode()
    manual_peptide = re.sub(r'\..*', '', manual_results[scan_index][4])

    # ignore mismatches between manual and OpenNuXL as non-conclusive
    if peptide != manual_peptide:    
      print("Mismatch (manual vs. search) " + str(scan_index) + ": "+ manual_peptide + "\t" + peptide + "\t" + str(score))
    else:
      print("Match (manual vs. search) " + str(scan_index) + ": " + manual_peptide + "\t" + peptide + "\t" + str(score))
      PSM2FDR[PSM] = score
      if peptide not in xl2minFDR.keys():
        xl2minFDR[peptide] = score
      else:
        if score < xl2minFDR[peptide]:
          xl2minFDR[peptide] = score
    previous_scan_index+=1

  # unidentified ones at the end
  while previous_scan_index <= n_manual_validated:
    #print("UNKNOWN;UNKNOWN;UNKNOWN;UNKNOWN")
    previous_scan_index+=1



#for k,v in xl2minFDR.items():
#  print(k+";"+str(v))
#print()
#for k,v in PSM2FDR.items():
#  print(k+";"+str(v))
#print(len(PSM2FDR))

  plt.title('XL-PSM FDR (' + filename + ')')
  qvalues = list(PSM2FDR.values())
  qvalues.sort()
  qvalues = [x * 100.0 for x in qvalues]

  r = range(1, 1+len(qvalues))
  r = [100.0 * float(x) / len(r) for x in r]

  print(qvalues)
  plt.plot(qvalues, r)
  plt.legend(loc = 'lower right')
  plt.xlim([0, 100.0])
  plt.ylim([0, 100])
  plt.ylabel('XL-PSMs (n=' + str(len(qvalues))  +  ') [%]')
  plt.xlabel('XL-PSM FDR cutoff [%]')
  plt.show()


exit()

