import difflib
import json
import os
import shutil
import xml.etree.ElementTree as ET
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from robosahaya.evidence_models import PatchProposal
from robosahaya.policy_gate import PolicyGate

class WorkspaceManager:
    """Manages isolated per-task run workspaces, golden templates, validation, and safe patching."""

    def __init__(self, base_workspace_root: Optional[Path] = None, templates_dir: Optional[Path] = None):
        self.base_root = (base_workspace_root or Path("/tmp/robosahaya_workspaces")).resolve()
        self.base_root.mkdir(parents=True, exist_ok=True)
        
        repo_root = Path(__file__).resolve().parent.parent.parent
        self.templates_dir = (templates_dir or (repo_root / "simulation" / "templates")).resolve()
        self.policy_gate = PolicyGate(self.base_root)

    def get_task_workspace(self, task_id: str) -> Path:
        """Returns the isolated workspace directory for a specific task."""
        clean_id = "".join(c for c in task_id if c.isalnum() or c in ("-", "_"))
        ws = (self.base_root / clean_id).resolve()
        if not self.policy_gate.validate_workspace_path(ws, self.base_root):
            raise ValueError(f"Task ID '{task_id}' results in path traversal outside workspace root.")
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    def assemble_simulation_template(
        self,
        task_id: str,
        template_id: str = "differential_drive",
        parameters: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Assembles a simulation workspace from verified golden templates using bounded parameters.
        """
        if template_id != "differential_drive":
            raise ValueError(f"Template '{template_id}' is not an approved golden template.")

        ws = self.get_task_workspace(task_id)
        source_dir = self.templates_dir / template_id

        if not source_dir.exists():
            raise FileNotFoundError(f"Template directory '{source_dir}' does not exist.")

        # Default bounded parameters
        p = {
            "robot_name": "sahaya_bot",
            "chassis_length": "0.45",
            "chassis_width": "0.35",
            "chassis_height": "0.15",
            "chassis_mass": "5.0",
            "wheel_radius": "0.08",
            "wheel_separation": "0.38",
            "lidar_rate": "10.0"
        }
        if parameters:
            p.update({k: str(v) for k, v in parameters.items()})

        # Copy bridge.yaml and robot.launch.py
        shutil.copy2(source_dir / "bridge.yaml", ws / "bridge.yaml")
        shutil.copy2(source_dir / "robot.launch.py", ws / "robot.launch.py")

        # Read xacro template, perform bounded parameter substitution
        with open(source_dir / "model.urdf.xacro", "r") as f:
            xacro_content = f.read()

        for key, val in p.items():
            xacro_content = xacro_content.replace(f"{{{{ {key} }}}}", val)

        # Save model.urdf
        urdf_dest = ws / "model.urdf"
        with open(urdf_dest, "w") as f:
            f.write(xacro_content)

        # Create manifest.json
        manifest = {
            "task_id": task_id,
            "template_id": template_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parameters": p,
            "files": ["model.urdf", "bridge.yaml", "robot.launch.py"]
        }
        with open(ws / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return ws

    def validate_workspace_files(self, workspace_dir: Path) -> Dict[str, Any]:
        """
        Validates XML and YAML syntax of generated workspace files.
        """
        results = {"valid": True, "checks": {}, "errors": []}

        # 1. Validate URDF XML
        urdf_file = workspace_dir / "model.urdf"
        if urdf_file.exists():
            try:
                ET.parse(str(urdf_file))
                results["checks"]["urdf_xml"] = "VALID"
            except Exception as e:
                results["valid"] = False
                results["checks"]["urdf_xml"] = f"INVALID: {str(e)}"
                results["errors"].append(str(e))
        else:
            results["valid"] = False
            results["checks"]["urdf_xml"] = "MISSING"

        # 2. Validate bridge.yaml
        bridge_file = workspace_dir / "bridge.yaml"
        if bridge_file.exists():
            try:
                with open(bridge_file, "r") as f:
                    yaml.safe_load(f)
                results["checks"]["bridge_yaml"] = "VALID"
            except Exception as e:
                results["valid"] = False
                results["checks"]["bridge_yaml"] = f"INVALID: {str(e)}"
                results["errors"].append(str(e))
        else:
            results["valid"] = False
            results["checks"]["bridge_yaml"] = "MISSING"

        return results

    def inject_demo_fault(self, workspace_dir: Path, fault_type: str = "missing_bridge_scan") -> PatchProposal:
        """
        Deterministically injects a known fault into the task workspace and returns the corresponding repair proposal.
        """
        target_file_rel = "bridge.yaml"
        bridge_file = workspace_dir / target_file_rel
        if not bridge_file.exists():
            raise FileNotFoundError("bridge.yaml missing in workspace.")

        with open(bridge_file, "r") as f:
            original_lines = f.readlines()

        if fault_type == "missing_bridge_scan":
            # Mutate /scan to /scan_disabled in bridge
            faulty_lines = [
                line.replace('ros_topic_name: "/scan"', 'ros_topic_name: "/scan_disabled"')
                for line in original_lines
            ]
            with open(bridge_file, "w") as f:
                f.writelines(faulty_lines)

            # Generate unified diff that will restore it
            diff = "".join(difflib.unified_diff(
                faulty_lines,
                original_lines,
                fromfile="a/" + target_file_rel,
                tofile="b/" + target_file_rel
            ))
            diff_hash = PolicyGate.compute_diff_sha256(diff)

            return PatchProposal(
                patch_id="patch_enable_laser_bridge",
                target_file=target_file_rel,
                unified_diff=diff,
                diff_sha256=diff_hash,
                explanation="Restore ROS 2 /scan topic mapping in ros_gz_bridge configuration",
                requires_approval=True,
                verification_test_id="test_laser_scan_payload",
                backup_required=True
            )

        raise ValueError(f"Unknown fault type: {fault_type}")

    def apply_reviewed_patch(
        self,
        workspace_dir: Path,
        proposal: PatchProposal,
        approval_token: str
    ) -> Dict[str, Any]:
        """
        Safely applies an approved patch with hash binding and backup creation.
        """
        target_path = workspace_dir / proposal.target_file
        if not target_path.exists():
            return {"success": False, "error": f"Target file '{proposal.target_file}' not found."}

        # Policy validation
        policy = PolicyGate(workspace_dir)
        decision = policy.verify_patch_approval(proposal, approval_token, proposal.unified_diff)
        if not decision.allowed:
            return {"success": False, "error": f"Security Policy Rejected: {decision.reason}"}

        # Create backup
        backup_path = target_path.with_suffix(target_path.suffix + ".bak")
        shutil.copy2(target_path, backup_path)

        try:
            # Apply patch lines
            with open(target_path, "r") as f:
                current_text = f.read()

            # Parse unified diff and apply patch
            lines = current_text.splitlines(keepends=True)
            patch_lines = proposal.unified_diff.splitlines(keepends=True)

            # Simple robust patch application for unified diff hunk
            for line in patch_lines:
                if line.startswith("-") and not line.startswith("---"):
                    target_sub = line[1:]
                    for add_line in patch_lines:
                        if add_line.startswith("+") and not add_line.startswith("+++"):
                            replace_with = add_line[1:]
                            current_text = current_text.replace(target_sub, replace_with)

            with open(target_path, "w") as f:
                f.write(current_text)

            return {
                "success": True,
                "backup_file": str(backup_path),
                "patched_file": str(target_path),
                "patch_id": proposal.patch_id
            }
        except Exception as ex:
            # Restore backup on failure
            if backup_path.exists():
                shutil.copy2(backup_path, target_path)
            return {"success": False, "error": f"Patch application failed: {str(ex)}"}
