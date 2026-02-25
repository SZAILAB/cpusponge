#include <cstdio>

#include "cpu_phase2_gates.h"

CONTROLLER controller;

int main(int argc, char* argv[]) {
    controller.Initial(argc, argv, "SPONGE CPU Phase-2 gatecheck");

    cpu_phase2::Check_Config_Gates(&controller);

    std::printf("CPU Phase-2 gatecheck passed: requested feature families are supported in scoped Phase 2.\n");
    return 0;
}
