import sys, os
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.permutation_test import (
    perm_test_HC_MDD,
    perm_test_slow_band,
    perm_test_atlas,
    perm_test_memd_bandpass,
)
from plots.plot_test_res import (
    plot_perm_test_result,
    plot_perm_test_results_grid,
    plot_perm_test_results_grid_AP_PA,
    plot_delta_atlas_comp,
    plot_combined_with_runs,
)
from src.fdr_correction import fdr_correct_pipeline
from src.data_loader import load_zFC_df

if __name__ == "__main__":

    # fdr_correct_pipeline()
    all_runs=True
    network_means = True
    decomp = "memd"
    # ^ options: "memd", "bandpass" ^
    use_fdr_pvals = False
    # plot_grids = [True, True, True]
    plot_grids = [False, False, False]

    which_atlases = ["Schaefer400", "Yan2023"]
    # ^ options: "Schaefer400", "Yan2023" ^
    which_freq_bands = ["full", "slow5", "slow4", "slow3"]
    # ^ options: "full", "slow5", "slow4" and "slow3" ^

    # which_perm_tests = ["hc_mdd", "band", "method", "atlas"]
    which_perm_tests = []
    # ^ options: "hc_mdd", "band", "method", "atlas" ^
    which_tasks = ["restAP", "restPA", "combined"]
    # which_tasks = ["combined"]
    # ^ options: "restAP", "restPA", "combined" ^
    
    which_pt_result_plots = ["hc_mdd", "band", "method", "atlas"]
    # which_pt_result_plots = ["atlas"]
    # ^ options: "hc_mdd", "band", "method", "atlas" ^
    which_combined_result_plots = ["hc_mdd", "band", "method", "atlas"]
    # which_combined_result_plots = ["atlas"]
    # ^ options: "hc_mdd", "band", "method", "atlas" ^

    for t in which_tasks:
        for band in tqdm(
            which_freq_bands,
            desc=f"Running permutation tests | task={t}",
            unit="band",
            leave=False,
        ):
            for a in which_atlases:
                if 'hc_mdd' in which_perm_tests:
                    print(f"Running HC-MDD permutation test for band: {band} (t: {t}, atlas: {a})")
                    perm_test_HC_MDD(
                        task_type=t,
                        atlas_type=a,
                        band_type=band,
                        network_means=network_means,
                        n_permutations=10_000,
                        test_dir='two_tailed',
                        decomp_method=decomp,
                        include_all_runs=all_runs,
                    )
                if band != "full":
                    if 'band' in which_perm_tests:
                        print(f"Running slow band comparison permutation test for band: {band} (t: {t}, atlas: {a})")
                        perm_test_slow_band(
                            task_type=t,
                            atlas_type=a,
                            slow_band=band,
                            network_means=network_means,
                            n_permutations=10_000,
                            test_dir='two_tailed',
                            decomp_method=decomp,
                            include_all_runs=all_runs,
                        )
                    if 'method' in which_perm_tests:
                        print(f"Running MEMD band-pass method comparison permutation test for band: {band} (t: {t}, atlas: {a})")
                        perm_test_memd_bandpass(
                            task_type=t,
                            atlas_type=a,
                            slow_band=band,
                            network_means=network_means,
                            n_permutations=10_000,
                            test_dir='two_tailed',
                            include_all_runs=all_runs,
                        )
                    if 'atlas' in which_perm_tests:
                        print(f"Running atlas comparison permutation test for band: {band} (t: {t})")
                        perm_test_atlas(
                            task_type=t,
                            band_type=band,
                            network_means=network_means,
                            n_permutations=10_000,
                            test_dir='two_tailed',
                            decomp_method=decomp,
                            include_all_runs=all_runs,
                        )


    for t in which_tasks:
        for band in tqdm(
            which_freq_bands,
            desc=f"Plotting permutation results | task={t}",
            unit="band",
            leave=False,
        ):
            for a in which_atlases:
                if 'hc_mdd' in which_pt_result_plots:
                    plot_perm_test_result(
                        test_type='hc_mdd',
                        task_type=t,
                        band_type=band,
                        atlas_type=a,
                        network_means=network_means,
                        decomp_method=decomp,
                        include_all_runs=all_runs,
                    )
                if band != "full":
                    if 'band' in which_pt_result_plots:
                        plot_perm_test_result(
                            test_type='band_comparison',
                            task_type=t,
                            band_type=band,
                            atlas_type=a,
                            network_means=network_means,
                            decomp_method=decomp,
                            include_all_runs=all_runs,
                        )
                    if 'method' in which_pt_result_plots:
                        plot_perm_test_result(
                            test_type='method_comparison',
                            task_type=t,
                            band_type=band,
                            atlas_type=a,
                            network_means=network_means,
                            decomp_method=decomp,
                            include_all_runs=all_runs,
                        )
                    if 'atlas' in which_pt_result_plots:
                        plot_perm_test_result(
                            test_type='atlas_comparison',
                            task_type=t,
                            band_type=band,
                            network_means=network_means,
                            decomp_method=decomp,
                            include_all_runs=all_runs,
                        )


    for a in which_atlases:
        for band in tqdm(
        which_freq_bands,
            desc=f"Plotting comparing plots | atlas={a}",
            unit="band",
            leave=False,
        ):
            if 'hc_mdd' in which_combined_result_plots:
                plot_combined_with_runs(
                    test_type='hc_mdd',
                    band_type=band,
                    atlas_type=a,
                    decomp_method=decomp,
                    use_fdr_pvals=use_fdr_pvals,
                    include_all_runs=all_runs,
                )
                if band != "full":
                    if 'band' in which_combined_result_plots:
                        plot_combined_with_runs(
                            test_type='band_comparison',
                            band_type=band,
                            atlas_type=a,
                            decomp_method=decomp,
                            use_fdr_pvals=use_fdr_pvals,
                            include_all_runs=all_runs,
                        )
                    if 'method' in which_combined_result_plots:
                        plot_combined_with_runs(
                            test_type='method_comparison',
                            band_type=band,
                            atlas_type=a,
                            decomp_method=decomp,
                            use_fdr_pvals=use_fdr_pvals,
                            include_all_runs=all_runs,
                        )
                    if 'atlas' in which_combined_result_plots:
                        plot_combined_with_runs(
                            test_type='atlas_comparison',
                            band_type=band,
                            atlas_type=a,
                            decomp_method=decomp,
                            use_fdr_pvals=use_fdr_pvals,
                            include_all_runs=all_runs,
                        )

        if plot_grids[0]:
            plot_perm_test_results_grid_AP_PA(
                atlas_type=a,
                network_means=network_means,
                decomp_method=decomp,
            )
        if plot_grids[1]:
            for task in which_tasks:
                plot_perm_test_results_grid(
                    task_type=t,
                    atlas_type=a,
                    network_means=network_means,
                    decomp_method=decomp,
                )
    if plot_grids[2]:
        for task in which_tasks:
            plot_delta_atlas_comp(
                task_type=t,
                network_means=network_means,
                decomp_method=decomp,
                include_all_runs=all_runs,
            )
                