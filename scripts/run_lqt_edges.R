args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NA_character_) {
  if (flag %in% args) {
    return(args[which(args == flag) + 1])
  }
  default
}

config_path <- get_arg("--config", Sys.getenv("NT_CONFIG", unset = "config/dat_config.yaml"))
config_path <- normalizePath(config_path, mustWork = TRUE)
project_guess <- dirname(dirname(config_path))
lib_dir <- file.path(project_guess, "external", "r_libs")
.libPaths(c(lib_dir, .libPaths()))

if (!requireNamespace("yaml", quietly = TRUE)) {
  stop("yaml is required. Run scripts/install_lqt_r_deps.R first.")
}
config <- yaml::read_yaml(config_path)
project_dir <- normalizePath(config$project_dir, mustWork = TRUE)
lib_dir <- file.path(project_dir, "external", "r_libs")
limit <- NA_integer_
if ("--limit" %in% args) {
  limit <- as.integer(args[which(args == "--limit") + 1])
} else if (!is.null(config$lqt$max_subjects)) {
  limit <- as.integer(config$lqt$max_subjects)
}
force <- "--force" %in% args

.libPaths(c(lib_dir, .libPaths()))
if (!requireNamespace("R.matlab", quietly = TRUE)) {
  stop("R.matlab is required. Run scripts/install_lqt_r_deps.R first.")
}

dsi_path <- file.path(project_dir, config$lqt$dsi_path)
Sys.setenv(NT_PROJECT_DIR = project_dir)
mount_root <- config$lqt$docker_mount_root
if (is.null(mount_root)) {
  mount_root <- dirname(dirname(project_dir))
}
Sys.setenv(NT_DOCKER_MOUNT_ROOT = mount_root)
source_path <- file.path(project_dir, config$lqt$data_dir, "Tractography_Atlas")
fib_path <- file.path(source_path, "HCP842_1mm.fib.gz")
tract_path <- file.path(source_path, "all_tracts_1mm.trk.gz")
atlas_path <- file.path(project_dir, config$atlases$outputs$atlas4s156_1mm_lqt)
edge_dir <- file.path(project_dir, config$outputs$edge_dir)
lqt_dir <- file.path(edge_dir, "lqt_2mm")
atlas_dir <- file.path(lqt_dir, "Atlas")
dir.create(edge_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(lqt_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(atlas_dir, recursive = TRUE, showWarnings = FALSE)

nodes <- read.csv(file.path(project_dir, config$atlases$outputs$atlas4s156_lqt_nodes))
manifest <- read.csv(file.path(project_dir, config$outputs$qc_dir, "subject_manifest.csv"))
dat_roi <- read.csv(file.path(project_dir, config$outputs$node_dir, "dat_roi_156.csv"))
dat_roi <- dat_roi[match(nodes$roi, dat_roi$roi), ]
dat_weight <- outer(dat_roi$dat_mean, dat_roi$dat_mean)
roi_count <- nrow(nodes)

if (!is.na(limit)) {
  manifest <- head(manifest, limit)
}

required <- c(dsi_path, fib_path, tract_path, atlas_path)
missing <- required[!file.exists(required)]
if (length(missing) > 0) {
  stop(paste("Missing required LQT input:", paste(missing, collapse = ", ")))
}

make_dsi_mni_link <- function(input_path, output_dir) {
  mni_name <- sub("\\.nii(\\.gz)?$", ".dsi.mni.nii.gz", basename(input_path))
  mni_path <- file.path(output_dir, mni_name)
  if (file.exists(mni_path) && !file.info(mni_path)$isdir) {
    unlink(mni_path)
  }
  if (!file.exists(mni_path)) {
    # 软链接保留原始2mm病灶，只用文件名标记MNI空间
    linked <- file.symlink(input_path, mni_path)
    if (!linked) {
      stop(paste("Could not create symlink for DSI input:", mni_path))
    }
  }
  mni_path
}

run_dsi <- function(output_dir, roi_path = NULL, allow_empty = FALSE) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  old_files <- list.files(output_dir, pattern = "connectivity\\.mat$", full.names = TRUE)
  old_times <- file.info(old_files)$mtime
  names(old_times) <- old_files
  roi_arg <- NULL
  if (!is.null(roi_path)) {
    # 让DSI明确按MNI空间读取病灶
    roi_arg <- make_dsi_mni_link(roi_path, output_dir)
  }
  dsi_args <- c(
    "--action=ana",
    paste0("--source=", fib_path),
    paste0("--tract=", tract_path),
    paste0("--output=", output_dir),
    paste0("--connectivity=", atlas_path),
    paste0("--connectivity_type=", config$lqt$con_type),
    "--connectivity_threshold=0"
  )
  if (!is.null(roi_arg)) {
    dsi_args <- c(dsi_args, paste0("--roi=", roi_arg))
  }
  # 调用DSI Studio计算LQT连接矩阵
  status <- system2(dsi_path, args = dsi_args, stdout = TRUE, stderr = TRUE)
  writeLines(status, file.path(output_dir, "dsi_studio.log"))
  # 修正Docker生成文件的权限
  system2("chmod", c("-R", "a+rwX", output_dir), stdout = FALSE, stderr = FALSE)
  all_files <- list.files(output_dir, pattern = "connectivity\\.mat$", full.names = TRUE)
  all_times <- file.info(all_files)$mtime
  changed <- all_files[!(all_files %in% names(old_times)) | all_times > old_times[all_files]]
  new_files <- changed[!is.na(changed)]
  if (length(new_files) == 0) {
    if (allow_empty && any(grepl("remaining tract count.*0|no tract", status, ignore.case = TRUE))) {
      return(NA_character_)
    }
    stop(paste("DSI Studio did not create a connectivity matrix in", output_dir))
  }
  info <- file.info(new_files)
  new_files[which.max(info$mtime)]
}

read_connectivity <- function(mat_path) {
  mat <- R.matlab::readMat(mat_path)
  mat_names <- names(mat)
  con_name <- NA_character_
  if ("connectivity" %in% mat_names) {
    con_name <- "connectivity"
  } else {
    candidates <- grep("number.*tracts.*r2r", mat_names, value = TRUE, ignore.case = TRUE)
    if (length(candidates) > 0) {
      con_name <- candidates[1]
    }
  }
  if (is.na(con_name)) {
    stop(paste("No region-to-region connectivity variable in", mat_path, "Available:", paste(mat_names, collapse = ", ")))
  }
  # 读取DSI Studio的区域间连接矩阵
  con <- mat[[con_name]]
  con <- matrix(as.numeric(con), nrow = roi_count, ncol = roi_count)
  con[is.na(con)] <- 0
  con
}

atlas_mat_path <- file.path(atlas_dir, "atlas4s156_connectivity.mat")
if (!file.exists(atlas_mat_path)) {
  mat_path <- run_dsi(atlas_dir)
  file.copy(mat_path, atlas_mat_path, overwrite = TRUE)
}
atlas_con <- read_connectivity(atlas_mat_path)
save(atlas_con, nodes, file = file.path(atlas_dir, "atlas4s156_connectivity.RData"))

edge_names <- c()
for (i in seq_len(roi_count - 1)) {
  for (j in (i + 1):roi_count) {
    edge_names <- c(edge_names, sprintf("edge_%03d_%03d", nodes$roi[i], nodes$roi[j]))
  }
}

raw_rows <- list()
dat_rows <- list()
for (row_index in seq_len(nrow(manifest))) {
  subject_id <- manifest$subject_id[row_index]
  lesion_path <- manifest$lesion_path[row_index]
  subject_dir <- file.path(lqt_dir, subject_id)
  out_matrix <- file.path(subject_dir, paste0(subject_id, "_lqt_sdc_matrix.csv"))
  if (force && file.exists(out_matrix)) {
    unlink(c(
      out_matrix,
      file.path(subject_dir, paste0(subject_id, "_lqt_disconnectivity.RData"))
    ))
  }
  if (!file.exists(out_matrix)) {
    mat_path <- run_dsi(subject_dir, lesion_path, allow_empty = TRUE)
    if (is.na(mat_path)) {
      # 没有streamline穿过病灶时记为零断连
      subject_con <- matrix(0, nrow = roi_count, ncol = roi_count)
    } else {
      subject_con <- read_connectivity(mat_path)
    }
    sdc <- subject_con / atlas_con
    sdc[is.na(sdc) | is.infinite(sdc)] <- 0
    write.csv(sdc, out_matrix, row.names = FALSE)
    save(subject_con, sdc, file = file.path(subject_dir, paste0(subject_id, "_lqt_disconnectivity.RData")))
  } else {
    sdc <- as.matrix(read.csv(out_matrix))
  }
  dat_edge <- sdc * dat_weight

  raw_values <- c(subject_id = subject_id)
  dat_values <- c(subject_id = subject_id)
  edge_index <- 1
  for (i in seq_len(roi_count - 1)) {
    for (j in (i + 1):roi_count) {
      raw_values[edge_names[edge_index]] <- sdc[i, j]
      dat_values[edge_names[edge_index]] <- dat_edge[i, j]
      edge_index <- edge_index + 1
    }
  }
  raw_rows[[row_index]] <- raw_values
  dat_rows[[row_index]] <- dat_values
  cat(sprintf("finished %s (%d/%d)\n", subject_id, row_index, nrow(manifest)))
}

raw_df <- as.data.frame(do.call(rbind, raw_rows), stringsAsFactors = FALSE)
dat_df <- as.data.frame(do.call(rbind, dat_rows), stringsAsFactors = FALSE)
for (col in edge_names) {
  raw_df[[col]] <- as.numeric(raw_df[[col]])
  dat_df[[col]] <- as.numeric(dat_df[[col]])
}
write.csv(raw_df, file.path(edge_dir, config$analysis$tables$lqt_edge_disconnection), row.names = FALSE)
write.csv(dat_df, file.path(edge_dir, config$analysis$tables$dat_edge_lqt), row.names = FALSE)
cat("wrote LQT edge tables\n")
