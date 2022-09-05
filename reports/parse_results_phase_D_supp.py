"""
File:
    reports/parse_results_phase_D_supp.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Parsing script for phase D supplementary results
"""

import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Opening JSON file
    with open('results_phase_D_supp.json') as json_file:
        results = json.load(json_file)

    valid = []
    ap = []
    ap50 = []

    for run_dict in dict(sorted(results.items())).values():
        valid.append(run_dict['args']['validation_size'])
        ap.append(run_dict['results']['AP @ [IoU=0.50:0.95 | area=all | maxDets=100]'])
        ap50.append(run_dict['results']['AP @ [IoU=0.50 | area=all | maxDets=100]'])

    err = 0.00747188

    print('AP List:')
    print([round(ap_ind, 4) for ap_ind in ap])
    print([round(ap_ind * err, 7) for ap_ind in ap])


    # print('AP_50 List:')
    # print([round(ap_ind, 4) for ap_ind in ap50])

    plt.errorbar(valid, ap, yerr=[ap_ind * err for ap_ind in ap], fmt='o-')
    plt.xlabel('Validation Size')
    plt.xlim([0.09, 0.51])
    plt.ylabel('AP')
    plt.ylim([0.7, 0.74])
    plt.tight_layout()
    plt.grid()
    plt.savefig('giard6.pdf')
    plt.show()

    plt.clf()

    # plt.errorbar(valid, ap50, yerr=[0.0076 / 2] * len(ap), fmt='o-', color='orange')
    # plt.xlabel('Validation Size')
    # plt.xlim([0.09, 0.51])
    # plt.ylabel('AP$_{50}$')
    # plt.ylim([0.895, 0.925])
    # plt.tight_layout()
    # plt.grid()
    # plt.savefig('AP_50.pdf')
    # plt.show()
