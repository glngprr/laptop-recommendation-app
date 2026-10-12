CREATE TABLE IF NOT EXISTS laptops (
  id                     INT UNSIGNED NOT NULL AUTO_INCREMENT,
  name                   VARCHAR(255) NOT NULL,
  price                  VARCHAR(32)  NULL,
  price_numeric          INT UNSIGNED NOT NULL,
  processor              VARCHAR(255) NULL,
  memory                 VARCHAR(128) NULL,
  ram_gb                 SMALLINT UNSIGNED NOT NULL,
  storage                VARCHAR(128) NULL,
  storage_gb             INT UNSIGNED NOT NULL,
  graphics               VARCHAR(255) NULL,
  cpu_cores              SMALLINT UNSIGNED NULL,
  gpu_cores              SMALLINT UNSIGNED NULL,
  url                    VARCHAR(255) NOT NULL,
  processor_normalized   VARCHAR(255) NULL,
  graphics_normalized    VARCHAR(255) NULL,
  cpu_passmark_mark      DOUBLE NULL,
  cpu_passmark_name      VARCHAR(128) NULL,
  cpu_mapping_status     VARCHAR(32)  NULL,
  gpu_passmark_g3d_mark  DOUBLE NULL,
  gpu_passmark_name      VARCHAR(128) NULL,
  gpu_mapping_status     VARCHAR(32)  NULL,
  benchmark_date         DATE NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_laptops_url (url),
  KEY idx_laptops_price (price_numeric)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS sessions (
  id                 CHAR(36) NOT NULL,
  created_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  profile            VARCHAR(32) NOT NULL,
  budget_idr         INT UNSIGNED NOT NULL,
  min_ram_gb         SMALLINT UNSIGNED NOT NULL,
  min_storage_gb     INT UNSIGNED NOT NULL,
  cpu_level          VARCHAR(32) NOT NULL,
  gpu_level          VARCHAR(32) NULL,
  require_dedicated  TINYINT(1) NOT NULL,
  lambda_surplus     DOUBLE NOT NULL,
  catalog_source     VARCHAR(16) NOT NULL,
  n_after_filter     INT NOT NULL,
  n_rankable         INT NOT NULL,
  n_recommendations  INT NOT NULL,
  comment            TEXT NULL,
  PRIMARY KEY (id),
  KEY idx_sessions_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS session_items (
  session_id    CHAR(36) NOT NULL,
  rank_position TINYINT UNSIGNED NOT NULL,
  laptop_id     INT UNSIGNED NOT NULL,
  knn_distance  DOUBLE NOT NULL,
  PRIMARY KEY (session_id, rank_position),
  UNIQUE KEY uq_session_laptop (session_id, laptop_id),
  CONSTRAINT fk_items_session FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE,
  CONSTRAINT fk_items_laptop  FOREIGN KEY (laptop_id)  REFERENCES laptops (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS feedback (
  id          INT UNSIGNED NOT NULL AUTO_INCREMENT,
  session_id  CHAR(36) NOT NULL,
  laptop_id   INT UNSIGNED NOT NULL,
  rating      TINYINT UNSIGNED NOT NULL,
  created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_feedback (session_id, laptop_id),
  CONSTRAINT chk_feedback_rating CHECK (rating BETWEEN 1 AND 5),
  CONSTRAINT fk_feedback_item FOREIGN KEY (session_id, laptop_id)
    REFERENCES session_items (session_id, laptop_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
