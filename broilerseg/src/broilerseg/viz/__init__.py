"""Visualisation module: every plot used by the notebooks lives here.

Sub-modules
-----------
style        shared colours and matplotlib style
overlays     masks on frames, per-class panels, galleries, bird selection
calibration  feeder measurement and pixel-to-cm stability
volume       alignment, slices, 3D cylinders, slice-count convergence, storyboard
dataset      video inventory, sampling plan, label statistics, YOLO dataset checks
mass_model   fitted mass equation, sensitivity, density estimate
metrics      weight accuracy, error analysis, density / slice sweeps, method comparison
training     YOLO training curves, teacher vs student agreement, speed
roi          stage 1 of two-stage labelling: pen regions and crops
"""

from broilerseg.viz.calibration import (
    plot_calibration_methods,
    plot_calibration_stability,
    plot_feeder_calibration,
)
from broilerseg.viz.dataset import (
    plot_contact_sheet,
    plot_folder_summary,
    plot_label_statistics,
    plot_motion_profile,
    plot_sampling_plan,
    plot_split_summary,
    plot_video_inventory,
    plot_weight_dataset_overview,
    plot_yolo_label_check,
)
from broilerseg.viz.mass_model import (
    plot_cv_landscape,
    plot_density_estimate,
    plot_fold_stability,
    plot_mask_adjustments,
    plot_mass_fit,
    plot_model_comparison,
    plot_nested_choices,
    plot_parameter_sensitivity,
)
from broilerseg.viz.metrics import (
    plot_accuracy_overview,
    plot_error_analysis,
    plot_feature_relationships,
    plot_frame_traces,
    plot_implied_density,
    plot_method_comparison,
    plot_per_event,
    plot_pred_vs_actual,
    plot_status_breakdown,
    plot_sweep_curves,
    plot_sweep_heatmap,
)
from broilerseg.viz.overlays import (
    blend_masks,
    plot_bird_selection,
    plot_gallery,
    plot_instances_by_prompt,
    plot_mask_cleanup,
    plot_overlay,
    plot_score_distributions,
    plot_segmentation_panels,
)
from broilerseg.viz.roi import (
    plot_pen_candidates,
    plot_pen_crops,
    plot_pen_rois,
    plot_roi_summary,
    plot_two_stage,
)
from broilerseg.viz.style import (
    fmt_weight,
    save,
    set_style,
    set_weight_unit,
    to_display,
    weight_factor,
    weight_table,
    weight_unit,
)
from broilerseg.viz.training import (
    plot_count_agreement,
    plot_iou_distribution,
    plot_speed_accuracy,
    plot_student_metrics,
    plot_teacher_student,
    plot_training_curves,
)
from broilerseg.viz.volume import (
    plot_alignment,
    plot_cylinders_3d,
    plot_slice_convergence,
    plot_slice_profile,
    plot_slices,
    plot_volume_storyboard,
)

__all__ = [
    "blend_masks",
    "plot_accuracy_overview",
    "plot_alignment",
    "plot_bird_selection",
    "plot_calibration_methods",
    "plot_calibration_stability",
    "plot_contact_sheet",
    "plot_count_agreement",
    "plot_cv_landscape",
    "plot_density_estimate",
    "plot_fold_stability",
    "plot_folder_summary",
    "plot_mask_adjustments",
    "plot_mass_fit",
    "plot_model_comparison",
    "plot_nested_choices",
    "plot_parameter_sensitivity",
    "plot_cylinders_3d",
    "plot_error_analysis",
    "plot_feature_relationships",
    "plot_feeder_calibration",
    "plot_frame_traces",
    "plot_gallery",
    "plot_implied_density",
    "plot_instances_by_prompt",
    "plot_iou_distribution",
    "plot_label_statistics",
    "plot_mask_cleanup",
    "plot_method_comparison",
    "plot_motion_profile",
    "plot_overlay",
    "plot_pen_candidates",
    "plot_pen_crops",
    "plot_pen_rois",
    "plot_per_event",
    "plot_roi_summary",
    "plot_two_stage",
    "plot_pred_vs_actual",
    "plot_sampling_plan",
    "plot_score_distributions",
    "plot_segmentation_panels",
    "plot_slice_convergence",
    "plot_slice_profile",
    "plot_slices",
    "plot_speed_accuracy",
    "plot_split_summary",
    "plot_status_breakdown",
    "plot_student_metrics",
    "plot_sweep_curves",
    "plot_sweep_heatmap",
    "plot_teacher_student",
    "plot_training_curves",
    "plot_video_inventory",
    "plot_volume_storyboard",
    "plot_weight_dataset_overview",
    "plot_yolo_label_check",
    "fmt_weight",
    "save",
    "set_style",
    "set_weight_unit",
    "to_display",
    "weight_factor",
    "weight_table",
    "weight_unit",
]
