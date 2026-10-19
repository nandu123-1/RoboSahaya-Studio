-- RoboSahaya Studio: Cloud Robot Model Library Schema Migration
-- Migration: 001_cloud_robot_library.sql
-- Target: PostgreSQL / Supabase

-- 1. Robot Models Table
CREATE TABLE IF NOT EXISTS robot_models (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    manufacturer TEXT DEFAULT 'Unknown',
    family TEXT NOT NULL,
    description TEXT DEFAULT '',
    aliases TEXT[] DEFAULT '{}',
    default_model_version TEXT DEFAULT 'v1.0.0',
    catalogue_status TEXT NOT NULL DEFAULT 'catalogued' CHECK (catalogue_status IN ('catalogued', 'registered', 'deprecated', 'unknown')),
    licence_identifier TEXT NOT NULL DEFAULT 'Check upstream source',
    licence_url TEXT DEFAULT '',
    metadata_json JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for searching and filtering by family
CREATE INDEX IF NOT EXISTS idx_robot_models_family ON robot_models(family);
CREATE INDEX IF NOT EXISTS idx_robot_models_status ON robot_models(catalogue_status);

-- 2. Robot Model Versions Table
CREATE TABLE IF NOT EXISTS robot_model_versions (
    id TEXT PRIMARY KEY,
    robot_model_id TEXT NOT NULL REFERENCES robot_models(id) ON DELETE CASCADE,
    version TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'upstream' CHECK (source_type IN ('bundled', 'upstream', 'managed_storage', 'git')),
    source_repository TEXT DEFAULT '',
    source_revision TEXT DEFAULT '',
    description_format TEXT NOT NULL CHECK (description_format IN ('URDF', 'XACRO', 'SDF', 'MJCF', 'XML')),
    entrypoint_path TEXT NOT NULL,
    manifest_path TEXT DEFAULT '',
    content_fingerprint TEXT NOT NULL,
    validation_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(robot_model_id, version)
);

CREATE INDEX IF NOT EXISTS idx_model_versions_robot ON robot_model_versions(robot_model_id);

-- 3. Robot Model Assets Table
CREATE TABLE IF NOT EXISTS robot_model_assets (
    id TEXT PRIMARY KEY,
    robot_model_version_id TEXT NOT NULL REFERENCES robot_model_versions(id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL,
    asset_type TEXT NOT NULL CHECK (asset_type IN ('mesh', 'texture', 'description', 'config', 'manifest', 'script')),
    object_storage_key TEXT DEFAULT '',
    upstream_reference TEXT DEFAULT '',
    sha256 TEXT NOT NULL,
    size_bytes BIGINT NOT NULL DEFAULT 0,
    licence_metadata JSONB DEFAULT '{}'::jsonb,
    required BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(robot_model_version_id, relative_path)
);

CREATE INDEX IF NOT EXISTS idx_model_assets_version ON robot_model_assets(robot_model_version_id);

-- 4. Simulator Adapters Table
CREATE TABLE IF NOT EXISTS simulator_adapters (
    id TEXT PRIMARY KEY,
    adapter_id TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    runtime_name TEXT NOT NULL,
    runtime_version_constraints TEXT DEFAULT '*',
    supported_description_formats TEXT[] NOT NULL,
    capability_metadata JSONB DEFAULT '{}'::jsonb,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. Robot Runtime Compatibility Table
CREATE TABLE IF NOT EXISTS robot_runtime_compatibility (
    id TEXT PRIMARY KEY,
    robot_model_version_id TEXT NOT NULL REFERENCES robot_model_versions(id) ON DELETE CASCADE,
    adapter_id TEXT NOT NULL REFERENCES simulator_adapters(adapter_id) ON DELETE CASCADE,
    simulator_version TEXT DEFAULT '',
    compatibility_status TEXT NOT NULL CHECK (compatibility_status IN ('compatible', 'incompatible_format', 'simulator_unavailable', 'unknown')),
    required_plugins TEXT[] DEFAULT '{}',
    required_controllers TEXT[] DEFAULT '{}',
    required_sensors TEXT[] DEFAULT '{}',
    required_bridges TEXT[] DEFAULT '{}',
    known_limitations TEXT DEFAULT '',
    last_validation_id TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(robot_model_version_id, adapter_id)
);

CREATE INDEX IF NOT EXISTS idx_runtime_compat_version ON robot_runtime_compatibility(robot_model_version_id);

-- 6. Robot Verification Runs Table
CREATE TABLE IF NOT EXISTS robot_verification_runs (
    id TEXT PRIMARY KEY,
    robot_model_version_id TEXT NOT NULL REFERENCES robot_model_versions(id) ON DELETE CASCADE,
    adapter_id TEXT NOT NULL REFERENCES simulator_adapters(adapter_id) ON DELETE CASCADE,
    world_id TEXT DEFAULT 'obstacle_world',
    test_suite_version TEXT NOT NULL DEFAULT 'v1.0.0',
    test_profile JSONB NOT NULL DEFAULT '{}'::jsonb,
    environment_fingerprint TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    overall_result TEXT NOT NULL CHECK (overall_result IN ('PASSED', 'FAILED', 'BLOCKED', 'SKIPPED')),
    check_results JSONB NOT NULL DEFAULT '[]'::jsonb,
    logs_reference TEXT DEFAULT '',
    report_fingerprint TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_verification_runs_version ON robot_verification_runs(robot_model_version_id);
CREATE INDEX IF NOT EXISTS idx_verification_runs_created ON robot_verification_runs(created_at DESC);

-- Enable Row Level Security (RLS)
ALTER TABLE robot_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE robot_model_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE robot_model_assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE simulator_adapters ENABLE ROW LEVEL SECURITY;
ALTER TABLE robot_runtime_compatibility ENABLE ROW LEVEL SECURITY;
ALTER TABLE robot_verification_runs ENABLE ROW LEVEL SECURITY;

-- Policies: Public Read Access (Anyone can browse models and verification records)
CREATE POLICY "Public Read robot_models" ON robot_models FOR SELECT USING (true);
CREATE POLICY "Public Read robot_model_versions" ON robot_model_versions FOR SELECT USING (true);
CREATE POLICY "Public Read robot_model_assets" ON robot_model_assets FOR SELECT USING (true);
CREATE POLICY "Public Read simulator_adapters" ON simulator_adapters FOR SELECT USING (true);
CREATE POLICY "Public Read robot_runtime_compatibility" ON robot_runtime_compatibility FOR SELECT USING (true);
CREATE POLICY "Public Read robot_verification_runs" ON robot_verification_runs FOR SELECT USING (true);

-- Policies: Authenticated / Service-Role Write Access
CREATE POLICY "Service Role All robot_models" ON robot_models FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service Role All robot_model_versions" ON robot_model_versions FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service Role All robot_model_assets" ON robot_model_assets FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service Role All simulator_adapters" ON simulator_adapters FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service Role All robot_runtime_compatibility" ON robot_runtime_compatibility FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service Role All robot_verification_runs" ON robot_verification_runs FOR ALL TO service_role USING (true) WITH CHECK (true);

-- Seed Simulator Adapters
INSERT INTO simulator_adapters (id, adapter_id, display_name, runtime_name, runtime_version_constraints, supported_description_formats, capability_metadata, enabled)
VALUES
    ('adapter_gazebo', 'gazebo_harmonic', 'Gazebo Harmonic Adapter', 'gz-sim', '>=8.0.0', ARRAY['URDF', 'XACRO', 'SDF'], '{"bridges": ["ros_gz_bridge"], "ros2": "jazzy"}'::jsonb, TRUE),
    ('adapter_mujoco', 'mujoco', 'MuJoCo Runtime Adapter', 'mujoco', '>=3.0.0', ARRAY['MJCF', 'XML'], '{"physics": ["euler", "rk4"], "passive_gui": true}'::jsonb, TRUE)
ON CONFLICT (adapter_id) DO NOTHING;

-- Seed Canonical Robot Models
INSERT INTO robot_models (id, display_name, manufacturer, family, description, aliases, licence_identifier, licence_url)
VALUES
    ('differential_drive', 'RoboSahaya Differential Drive', 'RoboSahaya', 'wheeled/mobile', 'Bundled verified differential-drive mobile robot with 2D LiDAR and camera.', ARRAY['diff drive', 'differential drive', 'wheeled base', 'sahaya bot', 'sahaya_bot'], 'Apache-2.0', 'https://www.apache.org/licenses/LICENSE-2.0'),
    ('franka_panda', 'Franka Emika Panda', 'Franka Emika', 'robotic arm', '7-DOF articulated research manipulator arm with parallel gripper and tabletop world.', ARRAY['franka panda', 'panda arm', 'panda robot', 'robotic_arm_6dof'], 'Apache-2.0', 'https://www.apache.org/licenses/LICENSE-2.0'),
    ('so_arm101', 'SO-ARM101 / SO-101', 'The Robot Studio', 'robotic arm', '3D-printable 6-DOF follower arm template with Feetech STS3215 bus servos.', ARRAY['so-101', 'so101', 'so arm 101', 'so-arm101', 'lerobot so101'], 'Apache-2.0', 'https://www.apache.org/licenses/LICENSE-2.0'),
    ('rsk', 'Robot Soccer Kit (RSK)', 'Google DeepMind', 'wheeled/mobile', 'MuJoCo Menagerie Robot Soccer Kit 4-wheel omnidirectional mobile soccer robot.', ARRAY['robot soccer kit', 'robot_soccer_kit', 'soccer robot'], 'Apache-2.0', 'https://github.com/google-deepmind/mujoco_menagerie/tree/main/robot_soccer_kit'),
    ('skydio_x2', 'Skydio X2', 'Skydio', 'drone/aerial', 'Autonomous quadcopter aerial vehicle (Rejection gate target).', ARRAY['skydio', 'skydio drone', 'x2 drone'], 'Apache-2.0', ''),
    ('so_arm100', 'SO-ARM100 / SO-100', 'The Robot Studio', 'robotic arm', '3D-printable 6-DOF arm description (on-demand catalogue target).', ARRAY['so-100', 'so100', 'so arm 100', 'so-arm100'], 'Apache-2.0', ''),
    ('ur5e', 'Universal Robots UR5e', 'Universal Robots', 'robotic arm', 'Industrial 6-DOF collaborative robotic arm description.', ARRAY['ur5', 'ur 5e', 'universal robots ur5e'], 'BSD-3-Clause', ''),
    ('unitree_go2', 'Unitree Go2', 'Unitree Robotics', 'quadruped', 'Quadruped robot description.', ARRAY['go2', 'unitree go2', 'go 2'], 'BSD-3-Clause', '')
ON CONFLICT (id) DO NOTHING;
