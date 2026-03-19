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
    load_perm_test_results,
    perm_test_slow_band,
    perm_test_atlas,
    perm_test_memd_bandpass,
)
from plot_test_res import (
    plot_perm_test_result,
    plot_perm_test_results_grid,
    plot_perm_test_results_grid_AP_PA,
    plot_delta_atlas_comp,
    plot_combined_with_runs,
)
from fdr_correction import fdr_correct_pipeline

if __name__ == "__main__":
    
    fdr_correct_pipeline()

    atlas = "Yan2023"
    task = "restAP"
    network_means = True
    decomp = "bandpass"
    use_fdr_pvals = not True
    plot_grids = [False, False, False]
    
    # plot_perm_test_results = ["hc_mdd", "band", "method", "atlas"]
    plot_perm_test_results = []
    plot_combined = ["hc_mdd", "band", "method", "atlas"]
    # ^ options: "hc_mdd", "band", "method", "atlas" ^

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
                if plot_grids[0]:
                    plot_perm_test_results_grid_AP_PA(
                        atlas_type=a,
                        network_means=network_means,
                        decomp_method=decomp,
                    )
                if plot_grids[1]:
                    plot_perm_test_results_grid(
                        task_type=t,
                        atlas_type=a,
                        network_means=network_means,
                        decomp_method=decomp,
                    )
                if 'hc_mdd' in plot_perm_test_results:
                    plot_perm_test_result(
                        test_type='hc_mdd',
                        task_type=t,
                        band_type=band,
                        atlas_type=a,
                        network_means=network_means,
                        decomp_method=decomp
                    )
                if 'hc_mdd' in plot_combined:
                    plot_combined_with_runs(
                        test_type='hc_mdd',
                        band_type=band,
                        atlas_type=a,
                        decomp_method=decomp,
                        use_fdr_pvals=use_fdr_pvals
                    )
                if band != "full":
                    if 'band' in plot_perm_test_results:
                        plot_perm_test_result(
                            test_type='band_comparison',
                            task_type=t,
                            band_type=band,
                            atlas_type=a,
                            network_means=network_means,
                            decomp_method=decomp
                        )
                    if 'band' in plot_combined:
                        plot_combined_with_runs(
                            test_type='band_comparison',
                            band_type=band,
                            atlas_type=a,
                            decomp_method=decomp,
                            use_fdr_pvals=use_fdr_pvals
                        )
                    if 'method' in plot_perm_test_results:
                        plot_perm_test_result(
                            test_type='method_comparison',
                            task_type=t,
                            band_type=band,
                            atlas_type=a,
                            network_means=network_means,
                            decomp_method=decomp
                        )
                    if 'method' in plot_combined:
                        plot_combined_with_runs(
                            test_type='method_comparison',
                            band_type=band,
                            atlas_type=a,
                            decomp_method=decomp,
                            use_fdr_pvals=use_fdr_pvals
                        )
            if 'atlas' in plot_perm_test_results:
                plot_perm_test_result(
                    test_type='atlas_comparison',
                    task_type=t,
                    band_type=band,
                    network_means=network_means,
                    decomp_method=decomp
                )
            if 'atlas' in plot_combined:
                plot_combined_with_runs(
                    test_type='atlas_comparison',
                    band_type=band,
                    atlas_type=a,
                    decomp_method=decomp,
                    use_fdr_pvals=use_fdr_pvals
                )
                plot_delta_atlas_comp(
                    task_type=t,
                    network_means=network_means,
                    decomp_method=decomp
                )
