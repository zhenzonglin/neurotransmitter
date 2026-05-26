args <- commandArgs(trailingOnly = TRUE)
project_dir <- Sys.getenv("NT_PROJECT_DIR", unset = getwd())
if ("--project-dir" %in% args) {
  project_dir <- args[which(args == "--project-dir") + 1]
}
lib_dir <- file.path(project_dir, "external", "r_libs")
dir.create(lib_dir, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib_dir, .libPaths()))

# 只安装LQT边分析需要的最小R依赖
packages <- c("R.matlab", "yaml")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  install.packages(missing, repos = "https://cloud.r-project.org", lib = lib_dir)
}

cat("R library paths:\n")
cat(.libPaths(), sep = "\n")
