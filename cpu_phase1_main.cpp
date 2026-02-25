#include <cstdio>

#include "cpu_phase1_gates.h"

CONTROLLER controller;

int main(int argc, char* argv[]) {
    controller.Initial(argc, argv, "SPONGE CPU Phase-1 gatecheck");

    cpu_phase1::Check_Config_Gates(&controller);

    std::printf("CPU Phase-1 gatecheck passed: no gated runtime JIT/plugin feature requested.\n");
    return 0;
}
