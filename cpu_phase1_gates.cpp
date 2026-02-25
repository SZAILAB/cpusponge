#include "cpu_phase1_gates.h"

namespace {

void ThrowGatedFeatureError(CONTROLLER* controller, const char* feature_id,
                                         const char* command_hint, const char* issue_hint) {
    std::string reason = "Reason:\n\tFeature ";
    reason += feature_id;
    reason += " is gated in CPU Phase 1.\n\tRequested by command: ";
    reason += command_hint;
    reason += "\n\tTrack: ";
    reason += issue_hint;
    controller->Throw_SPONGE_Error(spongeErrorNotImplemented, "cpu_phase1::Check_Config_Gates", reason.c_str());
}

}  // namespace

namespace cpu_phase1 {

void Check_Config_Gates(CONTROLLER* controller) {
    if (controller->Command_Exist("pairwise_force", "in_file")) {
        ThrowGatedFeatureError(controller, "GATE-JIT-PAIRWISE", "pairwise_force_in_file", "TBD");
    }
    if (controller->Command_Exist("listed_forces", "in_file")) {
        ThrowGatedFeatureError(controller, "GATE-JIT-LISTED", "listed_forces_in_file", "TBD");
    }
    if (controller->Command_Exist("soft_walls", "in_file")) {
        ThrowGatedFeatureError(controller, "GATE-JIT-SOFTWALL", "soft_walls_in_file", "TBD");
    }
    if (controller->Command_Exist("plugin")) {
        ThrowGatedFeatureError(controller, "GATE-PLUGIN-RUNTIME", "plugin", "TBD");
    }
}

}  // namespace cpu_phase1
