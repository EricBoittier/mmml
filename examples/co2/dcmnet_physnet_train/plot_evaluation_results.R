#!/usr/bin/env Rscript
"
Create comprehensive plots of evaluation results.

1D plots: bonds (r1, r2, r1+r2, r1*r2) and angle vs errors
2D plots: (r1+r2+r1r2, theta) vs errors

Usage:
    Rscript plot_evaluation_results.R <output_dir>
"

suppressPackageStartupMessages({
    library(ggplot2)
    library(dplyr)
    library(gridExtra)
    library(viridis)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
    stop("Usage: Rscript plot_evaluation_results.R <output_dir>")
}

output_dir <- args[1]
csv_file <- file.path(output_dir, "evaluation_results.csv")

if (!file.exists(csv_file)) {
    stop(paste("File not found:", csv_file))
}

cat("Loading data from", csv_file, "\n")
df <- read.csv(csv_file)

# Create plots directory
plots_dir <- file.path(output_dir, "plots")
dir.create(plots_dir, showWarnings = FALSE)

# Color scheme
colors <- c("train" = "#1f77b4", "valid" = "#ff7f0e", "test" = "#2ca02c")
shapes <- c("train" = 16, "valid" = 17, "test" = 18)

# Helper function to create 1D scatter plots
create_1d_plot <- function(df, x_var, y_var, title, y_label, output_file) {
    p <- ggplot(df, aes_string(x = x_var, y = y_var, color = "split", shape = "split")) +
        geom_point(alpha = 0.6, size = 0.8) +
        scale_color_manual(values = colors) +
        scale_shape_manual(values = shapes) +
        labs(
            title = title,
            x = switch(x_var,
                "r1" = "r1 (C-O1) [Å]",
                "r2" = "r2 (C-O2) [Å]",
                "r1_plus_r2" = "r1 + r2 [Å]",
                "r1_times_r2" = "r1 × r2 [Å²]",
                "angle" = "Angle (O1-C-O2) [°]",
                x_var
            ),
            y = y_label,
            color = "Split",
            shape = "Split"
        ) +
        theme_bw() +
        theme(
            plot.title = element_text(hjust = 0.5, face = "bold"),
            legend.position = "bottom"
        )
    
    ggsave(output_file, p, width = 8, height = 6, dpi = 300)
    return(p)
}

# Helper function to create 2D heatmap
create_2d_plot <- function(df, x_var, y_var, z_var, title, z_label, output_file) {
    library(ggplot2)
    
    # Filter valid data
    df_clean <- df %>%
        filter(!is.na(!!sym(z_var)), !is.na(!!sym(x_var)), !is.na(!!sym(y_var)))
    
    if (nrow(df_clean) == 0) {
        cat("  Warning: No valid data for", title, "\n")
        return(NULL)
    }
    
    # Create heatmap for each split
    plots <- list()
    for (split_name in unique(df_clean$split)) {
        df_split <- df_clean %>% filter(split == split_name)
        
        if (nrow(df_split) == 0) next
        
        # Use hexbin or tile plot
        # Try hexbin first (better for sparse data)
        tryCatch({
            p <- ggplot(df_split, aes_string(x = x_var, y = y_var, z = z_var)) +
                stat_summary_hex(fun = mean, bins = 30) +
                scale_fill_viridis_c(name = z_label) +
                labs(
                    title = paste(title, "-", split_name),
                    x = switch(x_var,
                        "r1_plus_r2_plus_r1r2" = "r1 + r2 + r1×r2 [Å + Å²]",
                        "r1_plus_r2" = "r1 + r2 [Å]",
                        x_var
                    ),
                    y = switch(y_var,
                        "angle" = "Angle (O1-C-O2) [°]",
                        y_var
                    )
                ) +
                theme_bw() +
                theme(plot.title = element_text(hjust = 0.5, face = "bold"))
            
            plots[[split_name]] <- p
        }, error = function(e) {
            # Fallback to scatter with color mapping
            p <- ggplot(df_split, aes_string(x = x_var, y = y_var, color = z_var)) +
                geom_point(alpha = 0.6, size = 1) +
                scale_color_viridis_c(name = z_label) +
                labs(
                    title = paste(title, "-", split_name),
                    x = switch(x_var,
                        "r1_plus_r2_plus_r1r2" = "r1 + r2 + r1×r2 [Å + Å²]",
                        "r1_plus_r2" = "r1 + r2 [Å]",
                        x_var
                    ),
                    y = switch(y_var,
                        "angle" = "Angle (O1-C-O2) [°]",
                        y_var
                    )
                ) +
                theme_bw() +
                theme(plot.title = element_text(hjust = 0.5, face = "bold"))
            
            plots[[split_name]] <<- p
        })
    }
    
    if (length(plots) == 0) {
        cat("  Warning: No plots created for", title, "\n")
        return(NULL)
    }
    
    # Combine plots
    if (length(plots) > 1) {
        combined <- do.call(grid.arrange, c(plots, ncol = min(3, length(plots))))
    } else {
        combined <- plots[[1]]
    }
    
    ggsave(output_file, combined, width = 6 * length(plots), height = 5, dpi = 300)
    
    return(combined)
}

cat("\n=== Creating 1D Plots ===\n")

# 1. Energy errors
cat("  Energy errors...\n")
create_1d_plot(
    df, "r1", "energy_abs_error",
    "Energy Error vs r1", "|Energy Error| [eV]",
    file.path(plots_dir, "energy_error_vs_r1.png")
)
create_1d_plot(
    df, "r2", "energy_abs_error",
    "Energy Error vs r2", "|Energy Error| [eV]",
    file.path(plots_dir, "energy_error_vs_r2.png")
)
create_1d_plot(
    df, "r1_plus_r2", "energy_abs_error",
    "Energy Error vs r1 + r2", "|Energy Error| [eV]",
    file.path(plots_dir, "energy_error_vs_r1_plus_r2.png")
)
create_1d_plot(
    df, "r1_times_r2", "energy_abs_error",
    "Energy Error vs r1 × r2", "|Energy Error| [eV]",
    file.path(plots_dir, "energy_error_vs_r1_times_r2.png")
)
create_1d_plot(
    df, "angle", "energy_abs_error",
    "Energy Error vs Angle", "|Energy Error| [eV]",
    file.path(plots_dir, "energy_error_vs_angle.png")
)

# 2. Force norm errors
cat("  Force norm errors...\n")
create_1d_plot(
    df, "r1", "force_norm_abs_error",
    "Force Norm Error vs r1", "|Force Norm Error| [eV/Å]",
    file.path(plots_dir, "force_norm_error_vs_r1.png")
)
create_1d_plot(
    df, "r2", "force_norm_abs_error",
    "Force Norm Error vs r2", "|Force Norm Error| [eV/Å]",
    file.path(plots_dir, "force_norm_error_vs_r2.png")
)
create_1d_plot(
    df, "r1_plus_r2", "force_norm_abs_error",
    "Force Norm Error vs r1 + r2", "|Force Norm Error| [eV/Å]",
    file.path(plots_dir, "force_norm_error_vs_r1_plus_r2.png")
)
create_1d_plot(
    df, "r1_times_r2", "force_norm_abs_error",
    "Force Norm Error vs r1 × r2", "|Force Norm Error| [eV/Å]",
    file.path(plots_dir, "force_norm_error_vs_r1_times_r2.png")
)
create_1d_plot(
    df, "angle", "force_norm_abs_error",
    "Force Norm Error vs Angle", "|Force Norm Error| [eV/Å]",
    file.path(plots_dir, "force_norm_error_vs_angle.png")
)

# 3. Force max errors
cat("  Force max errors...\n")
create_1d_plot(
    df, "r1", "force_max_abs_error",
    "Force Max Error vs r1", "Max |Force Error| [eV/Å]",
    file.path(plots_dir, "force_max_error_vs_r1.png")
)
create_1d_plot(
    df, "r2", "force_max_abs_error",
    "Force Max Error vs r2", "Max |Force Error| [eV/Å]",
    file.path(plots_dir, "force_max_error_vs_r2.png")
)
create_1d_plot(
    df, "angle", "force_max_abs_error",
    "Force Max Error vs Angle", "Max |Force Error| [eV/Å]",
    file.path(plots_dir, "force_max_error_vs_angle.png")
)

# 4. Dipole (PhysNet) errors
cat("  Dipole (PhysNet) errors...\n")
create_1d_plot(
    df, "r1", "dipole_physnet_norm_abs_error",
    "Dipole (PhysNet) Norm Error vs r1", "|Dipole Norm Error| [e·Å]",
    file.path(plots_dir, "dipole_physnet_error_vs_r1.png")
)
create_1d_plot(
    df, "r2", "dipole_physnet_norm_abs_error",
    "Dipole (PhysNet) Norm Error vs r2", "|Dipole Norm Error| [e·Å]",
    file.path(plots_dir, "dipole_physnet_error_vs_r2.png")
)
create_1d_plot(
    df, "angle", "dipole_physnet_norm_abs_error",
    "Dipole (PhysNet) Norm Error vs Angle", "|Dipole Norm Error| [e·Å]",
    file.path(plots_dir, "dipole_physnet_error_vs_angle.png")
)

# 5. Dipole (DCMNet) errors
cat("  Dipole (DCMNet) errors...\n")
create_1d_plot(
    df, "r1", "dipole_dcmnet_norm_abs_error",
    "Dipole (DCMNet) Norm Error vs r1", "|Dipole Norm Error| [e·Å]",
    file.path(plots_dir, "dipole_dcmnet_error_vs_r1.png")
)
create_1d_plot(
    df, "r2", "dipole_dcmnet_norm_abs_error",
    "Dipole (DCMNet) Norm Error vs r2", "|Dipole Norm Error| [e·Å]",
    file.path(plots_dir, "dipole_dcmnet_error_vs_r2.png")
)
create_1d_plot(
    df, "angle", "dipole_dcmnet_norm_abs_error",
    "Dipole (DCMNet) Norm Error vs Angle", "|Dipole Norm Error| [e·Å]",
    file.path(plots_dir, "dipole_dcmnet_error_vs_angle.png")
)

# 6. ESP errors
cat("  ESP errors...\n")
df_esp <- df %>% filter(!is.na(esp_rmse_physnet))
if (nrow(df_esp) > 0) {
    create_1d_plot(
        df_esp, "r1", "esp_rmse_physnet",
        "ESP RMSE (PhysNet) vs r1", "ESP RMSE [Ha/e]",
        file.path(plots_dir, "esp_rmse_physnet_vs_r1.png")
    )
    create_1d_plot(
        df_esp, "r2", "esp_rmse_physnet",
        "ESP RMSE (PhysNet) vs r2", "ESP RMSE [Ha/e]",
        file.path(plots_dir, "esp_rmse_physnet_vs_r2.png")
    )
    create_1d_plot(
        df_esp, "angle", "esp_rmse_physnet",
        "ESP RMSE (PhysNet) vs Angle", "ESP RMSE [Ha/e]",
        file.path(plots_dir, "esp_rmse_physnet_vs_angle.png")
    )
    
    create_1d_plot(
        df_esp, "r1", "esp_rmse_dcmnet",
        "ESP RMSE (DCMNet) vs r1", "ESP RMSE [Ha/e]",
        file.path(plots_dir, "esp_rmse_dcmnet_vs_r1.png")
    )
    create_1d_plot(
        df_esp, "r2", "esp_rmse_dcmnet",
        "ESP RMSE (DCMNet) vs r2", "ESP RMSE [Ha/e]",
        file.path(plots_dir, "esp_rmse_dcmnet_vs_r2.png")
    )
    create_1d_plot(
        df_esp, "angle", "esp_rmse_dcmnet",
        "ESP RMSE (DCMNet) vs Angle", "ESP RMSE [Ha/e]",
        file.path(plots_dir, "esp_rmse_dcmnet_vs_angle.png")
    )
}

cat("\n=== Creating 2D Plots ===\n")

# Create r1+r2+r1r2 feature for 2D plots
df <- df %>% mutate(
    r1_plus_r2_plus_r1r2 = r1_plus_r2 + r1_times_r2
)

# 1. Energy errors
cat("  Energy errors (2D)...\n")
create_2d_plot(
    df, "r1_plus_r2_plus_r1r2", "angle", "energy_abs_error",
    "Energy Error", "|Energy Error| [eV]",
    file.path(plots_dir, "energy_error_2d.png")
)

# 2. Force norm errors
cat("  Force norm errors (2D)...\n")
create_2d_plot(
    df, "r1_plus_r2_plus_r1r2", "angle", "force_norm_abs_error",
    "Force Norm Error", "|Force Norm Error| [eV/Å]",
    file.path(plots_dir, "force_norm_error_2d.png")
)

# 3. Force max errors
cat("  Force max errors (2D)...\n")
create_2d_plot(
    df, "r1_plus_r2_plus_r1r2", "angle", "force_max_abs_error",
    "Force Max Error", "Max |Force Error| [eV/Å]",
    file.path(plots_dir, "force_max_error_2d.png")
)

# 4. Dipole (PhysNet) errors
cat("  Dipole (PhysNet) errors (2D)...\n")
create_2d_plot(
    df, "r1_plus_r2_plus_r1r2", "angle", "dipole_physnet_norm_abs_error",
    "Dipole (PhysNet) Norm Error", "|Dipole Norm Error| [e·Å]",
    file.path(plots_dir, "dipole_physnet_error_2d.png")
)

# 5. Dipole (DCMNet) errors
cat("  Dipole (DCMNet) errors (2D)...\n")
create_2d_plot(
    df, "r1_plus_r2_plus_r1r2", "angle", "dipole_dcmnet_norm_abs_error",
    "Dipole (DCMNet) Norm Error", "|Dipole Norm Error| [e·Å]",
    file.path(plots_dir, "dipole_dcmnet_error_2d.png")
)

# 6. ESP errors
if (nrow(df_esp) > 0) {
    cat("  ESP errors (2D)...\n")
    df_esp_2d <- df_esp %>% mutate(
        r1_plus_r2_plus_r1r2 = r1_plus_r2 + r1_times_r2
    )
    create_2d_plot(
        df_esp_2d, "r1_plus_r2_plus_r1r2", "angle", "esp_rmse_physnet",
        "ESP RMSE (PhysNet)", "ESP RMSE [Ha/e]",
        file.path(plots_dir, "esp_rmse_physnet_2d.png")
    )
    create_2d_plot(
        df_esp_2d, "r1_plus_r2_plus_r1r2", "angle", "esp_rmse_dcmnet",
        "ESP RMSE (DCMNet)", "ESP RMSE [Ha/e]",
        file.path(plots_dir, "esp_rmse_dcmnet_2d.png")
    )
}

cat("\n=== DONE ===\n")
cat("Plots saved to:", plots_dir, "\n")

