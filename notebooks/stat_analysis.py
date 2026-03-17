import sys, os
from tqdm import tqdm

PROJECT_ROOT = "/cluster/home/oystkva/project/code"
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "plots"))
from config import (
    DATA_DIR,
    LOG_DIR,
)
from permutation_test import (
    perm_test_HC_MDD,
    plot_perm_test_result,
    load_perm_test_results,
    plot_perm_test_results_grid,
    plot_perm_test_results_grid2,
    plot_delta_atlas_comp,
    plot_combined_with_runs,
    perm_test_slow_band,
    perm_test_atlas,
    perm_test_memd_bandpass,
)
from fdr_correction import fdr_correct_pipeline

if __name__ == "__main__":
    
    fdr_correct_pipeline()

    atlas = "Yan2023"
    task = "restAP"
    network_means = True
    decomp = "bandpass"
    use_fdr_pvals = not True

    # for t in ["combined"]:
    #     for band in ["full", "slow5", "slow4", "slow3"]:
    #         for a in ["Yan2023", "Schaefer400"]:
    #             print(f"Running HC-MDD permutation test for band: {band} (t: {t}, atlas: {a})")
    #             perm_test_HC_MDD(
    #                 task_type=t,
    #                 atlas_type=a,
    #                 band_type=band,
    #                 network_means=network_means,
    #                 n_permutations=10_000,
    #                 test_dir='two_tailed',
    #                 decomp_method=decomp
    #             )
    #             if band == "full":
    #                 continue
    #             for a in ["Yan2023", "Schaefer400"]:
    #                 print(f"Running slow band comparison permutation test for band: {band} (t: {t}, atlas: {a})")
    #                 perm_test_slow_band(
    #                     task_type=t,
    #                     atlas_type=a,
    #                     slow_band=band,
    #                     network_means=network_means,
    #                     n_permutations=10_000,
    #                     test_dir='two_tailed',
    #                     decomp_method=decomp
    #                 )
    #                 print(f"Running MEMD band-pass method comparison permutation test for band: {band} (t: {t}, atlas: {a})")
    #                 perm_test_memd_bandpass(
    #                     task_type=t,
    #                     atlas_type=a,
    #                     slow_band=band,
    #                     network_means=network_means,
    #                     n_permutations=10_000,
    #                     test_dir='two_tailed'
    #                 )
    #             print(f"Running atlas comparison permutation test for band: {band} (t: {t})")
    #             perm_test_atlas(
    #                 task_type=t,
    #                 band_type=band,
    #                 network_means=network_means,
    #                 n_permutations=10_000,
    #                 test_dir='two_tailed',
    #                 decomp_method=decomp
    #             )



    for t in ["combined"]:
        for band in tqdm(["full", "slow5", "slow4", "slow3"]):
            for a in ["Yan2023"]:
                # plot_perm_test_results_grid2(
                #     atlas_type=a,
                #     network_means=network_means,
                #     decomp_method=decomp,
                # )
                # plot_perm_test_results_grid(
                #     task_type=t,
                #     atlas_type=a,
                #     network_means=network_means,
                #     decomp_method=decomp,
                # )
                # plot_perm_test_result(
                #     test_type='hc_mdd',
                #     task_type=t,
                #     band_type=band,
                #     atlas_type=a,
                #     network_means=network_means,
                #     decomp_method=decomp
                # )
                # plot_combined_with_runs(
                #     test_type='hc_mdd',
                #     band_type=band,
                #     atlas_type=a,
                #     decomp_method=decomp,
                #     use_fdr_pvals=use_fdr_pvals
                # )
                if band != "full":
                #     plot_perm_test_result(
                #         test_type='band_comparison',
                #         task_type=t,
                #         band_type=band,
                #         atlas_type=a,
                #         network_means=network_means,
                #         decomp_method=decomp
                #     )
                    # plot_combined_with_runs(
                    #     test_type='band_comparison',
                    #     band_type=band,
                    #     atlas_type=a,
                    #     decomp_method=decomp,
                    #     use_fdr_pvals=use_fdr_pvals
                    # )
            #         plot_perm_test_result(
            #             test_type='method_comparison',
            #             task_type=t,
            #             band_type=band,
            #             atlas_type=a,
            #             network_means=network_means,
            #             decomp_method=decomp
            #         )
                    plot_combined_with_runs(
                        test_type='method_comparison',
                        band_type=band,
                        atlas_type=a,
                        decomp_method=decomp,
                        use_fdr_pvals=use_fdr_pvals
                    )
            # plot_perm_test_result(
            #     test_type='atlas_comparison',
            #     task_type=t,
            #     band_type=band,
            #     network_means=network_means,
            #     decomp_method=decomp
            # )
            # plot_combined_with_runs(
            #     test_type='atlas_comparison',
            #     band_type=band,
            #     atlas_type=a,
            #     decomp_method=decomp,
            #     use_fdr_pvals=use_fdr_pvals
            # )
            # plot_delta_atlas_comp(
            #     task_type=t,
            #     network_means=network_means,
            #     decomp_method=decomp
            # )
