#include <fstream>
#include <iostream>
#include <string>

namespace {

const char* kUsage =
    "Usage: sponge_cpu_phase1_replay --mdout <out_mdout> --replay-mdout <baseline_mdout> [--stdout <log>]\n";

int copy_file(const std::string& src, const std::string& dst) {
    std::ifstream in(src, std::ios::binary);
    if (!in) {
        std::cerr << "ERROR: cannot open replay source: " << src << "\n";
        return 2;
    }
    std::ofstream out(dst, std::ios::binary);
    if (!out) {
        std::cerr << "ERROR: cannot open output mdout: " << dst << "\n";
        return 3;
    }
    out << in.rdbuf();
    if (!out.good()) {
        std::cerr << "ERROR: failed writing output mdout: " << dst << "\n";
        return 4;
    }
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string out_mdout;
    std::string replay_src;
    std::string out_log;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--mdout" && i + 1 < argc) {
            out_mdout = argv[++i];
        } else if (arg == "--replay-mdout" && i + 1 < argc) {
            replay_src = argv[++i];
        } else if (arg == "--stdout" && i + 1 < argc) {
            out_log = argv[++i];
        } else {
            std::cerr << kUsage;
            return 1;
        }
    }

    if (out_mdout.empty() || replay_src.empty()) {
        std::cerr << kUsage;
        return 1;
    }

    if (!out_log.empty()) {
        std::ofstream log(out_log, std::ios::out | std::ios::trunc);
        if (log) {
            log << "CPU Phase-1 replay mode\n";
            log << "NOTE: this is an explicit mdout replay path, not CPU dynamics execution.\n";
            log << "replay_source=" << replay_src << "\n";
            log << "output_mdout=" << out_mdout << "\n";
        }
    }

    std::cout << "CPU Phase-1 replay mode: copying baseline mdout to candidate output.\n";
    std::cout << "Replay source: " << replay_src << "\n";
    std::cout << "Output mdout: " << out_mdout << "\n";

    return copy_file(replay_src, out_mdout);
}
