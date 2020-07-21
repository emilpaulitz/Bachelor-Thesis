import csv
import pyopenms
import re
import sys
import pprint
from pyopenms import *
import pandas as pd
import matplotlib.pyplot as plt



# load manual validated results
#PEPTIDE_COLUMN = 6
#for filename in {"A_Stuetzer_180816_12mer_hits.csv"}: # csv filenames
PEPTIDE_COLUMN = 2
for filename in {"A_Stuetzer_180816_12mer_Welp.csv"}: # csv filenames
  with open(filename) as csv_file:
    manual_results = list(csv.reader(csv_file, delimiter=';'))
manual_results.pop(0) # remove header
n_manual_validated = len(manual_results)

print(n_manual_validated)

# load identification results
prot_ids = []; pep_ids = []
#IdXMLFile().load("entrapment_perc_1.0000_XLs.idXML", prot_ids, pep_ids)
#IdXMLFile().load("entrapment_Welp.idXML_nuxl.idXML", prot_ids, pep_ids)
IdXMLFile().load(sys.argv[1], prot_ids, pep_ids)

xl2minFDR = dict()
PSM2FDR = dict()

previous_scan_index = 0

# Iterate over all protein hits
for peptide_id in pep_ids:
  scan_index = peptide_id.getMetaValue("scan_index")
  if scan_index > n_manual_validated:
    break

  while previous_scan_index < scan_index: #output lines for unidentified PSMs
    #print("UNKNOWN;UNKNOWN;UNKNOWN;UNKNOWN")
    previous_scan_index+=1

  hit = peptide_id.getHits()[0]
  score = hit.getScore()
  PSM = hit.getSequence().toString().decode() + "_" + str(int(peptide_id.getRT()))
  #print(str(peptide_id.getRT())+";"+str(peptide_id.getMZ())+";"+hit.getSequence().toString().decode()+";"+hit.getMetaValue("NuXL:NA").decode()+";"+str(hit.getScore()))
  peptide = hit.getSequence().toUnmodifiedString().decode()
  manual_peptide = re.sub(r'\..*', '', manual_results[scan_index][PEPTIDE_COLUMN])

  # ignore mismatches between manual and OpenNuXL as non-conclusive
  if peptide != manual_peptide:    
    print(manual_peptide + "\t" + peptide + "\t" + str(score))
  else:
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

plt.title('XL-PSM FDR')
qvalues = list(PSM2FDR.values())
qvalues.sort()
qvalues = [x * 100.0 for x in qvalues]

r = range(1, 1+len(qvalues))
r = [100.0 * float(x) / len(r) for x in r]

print(qvalues)
plt.plot(qvalues, r)
plt.legend(loc = 'lower right')
plt.xlim([0, 5.0])
plt.ylim([0, 100])
plt.ylabel('XL-PSMs (n=' + str(len(qvalues))  +  ') [%]')
plt.xlabel('XL-PSM FDR cutoff [%]')
plt.show()


exit()

from collections import defaultdict
rt_mz2indices_dict = defaultdict(list) # default value type of dictionary is a list
    
# build dictionary form filename to spectra
line_count = 0
for filename in {"A_Stuetzer_180816_12mer_hits.csv"}: # csv filenames
  with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    for row in csv_reader:
      if line_count == 0:  # skip header
        line_count += 1
        continue
      filename = row[0]
      index = int(row[1])
      file2indices_dict[filename].append(index)

print(file2indices_dict)

current_filename = ""
mzmlfile = MzMLFile()
spectra = MSExperiment()

for mzml_filename, indices in file2indices_dict.items():
    mzmlfile.load(mzml_filename, spectra) # load new (unfiltered) file
    print(mzml_filename)
    print(len(indices))
    filtered_spectra = MSExperiment()
    for index in indices:
      filtered_spectra.addSpectrum(spectra[index])
    mzmlfile.store(mzml_filename + "_filtered.mzML", filtered_spectra)

