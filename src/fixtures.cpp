#include "fixtures.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFont>
#include <QPainter>
#include <QPainterPath>
#include <QPdfWriter>
#include <QtEndian>

#include <cmath>

namespace studio {

QString fixtureDir()
{
    const QString dir = QCoreApplication::applicationDirPath() + QStringLiteral("/fixtures");
    QDir().mkpath(dir);
    return dir;
}

namespace {

QString specPdfPath() { return fixtureDir() + QStringLiteral("/FP4_decode_spec.pdf"); }
QString wavPath() { return fixtureDir() + QStringLiteral("/fn_key_click.wav"); }
QString svgPath() { return fixtureDir() + QStringLiteral("/frame_pacing.svg"); }

void writeSpecPdf(const QString &path)
{
    QPdfWriter pdf(path);
    pdf.setPageSize(QPageSize(QPageSize::A4));
    pdf.setPageMargins(QMarginsF(0, 0, 0, 0));
    pdf.setResolution(144);
    pdf.setTitle(QStringLiteral("FP4 KV-cache decode on sm120"));
    pdf.setCreator(QStringLiteral("lapis_studio fixtures"));
    QPainter p(&pdf);
    const int W = pdf.width();
    const int M = 110;
    QFont title(QStringLiteral("Avenir Next"));
    title.setPixelSize(44);
    title.setWeight(QFont::DemiBold);
    QFont body(QStringLiteral("Avenir Next"));
    body.setPixelSize(21);
    QFont mono(QStringLiteral("Menlo"));
    mono.setPixelSize(18);

    p.setPen(QColor(20, 20, 20));
    p.setFont(title);
    p.drawText(M, 150, QStringLiteral("FP4 KV-cache decode"));
    p.setFont(body);
    p.setPen(QColor(90, 90, 90));
    p.drawText(M, 196, QStringLiteral("sm120 · batch 1 · 32k context · spec v3"));

    p.setPen(QColor(20, 20, 20));
    const QStringList para = {
        QStringLiteral("Decode is bandwidth bound at batch 1. The KV cache is stored as"),
        QStringLiteral("packed FP4 with one FP8 scale per 32 values; dequant happens in"),
        QStringLiteral("registers right before the QK dot product, so no unpacked copy"),
        QStringLiteral("ever touches shared memory."),
    };
    int y = 280;
    for (const QString &l : para) {
        p.drawText(M, y, l);
        y += 34;
    }

    // Figure: four arms, gold winner.
    y += 40;
    p.setFont(body);
    p.drawText(M, y, QStringLiteral("Figure 1. Decode throughput by arm (tok/s)"));
    y += 30;
    const double vals[4] = {11.25, 17.9, 22.4, 29.6};
    const char *names[4] = {"baseline", "128x256", "dcp4-split", "128-reg"};
    const int barH = 46;
    const int maxW = W - 2 * M - 260;
    p.setFont(mono);
    for (int i = 0; i < 4; ++i) {
        const int by = y + i * (barH + 18);
        p.setPen(QColor(40, 40, 40));
        p.drawText(M, by + 31, QString::fromLatin1(names[i]));
        const int bw = int(maxW * vals[i] / 29.6);
        p.fillRect(M + 170, by, bw, barH, i == 3 ? QColor(0xe0, 0x9f, 0x3e) : QColor(0xb5, 0x65, 0x3b));
        p.setPen(QColor(40, 40, 40));
        p.drawText(M + 180 + bw, by + 31, QString::number(vals[i], 'f', vals[i] < 12 ? 2 : 1));
    }
    y += 4 * (barH + 18) + 60;

    p.setFont(body);
    p.drawText(M, y, QStringLiteral("Table 1. Launch configuration"));
    y += 40;
    p.setFont(mono);
    const QStringList rows = {
        QStringLiteral("arm         tile     regs  spills  tok/s"),
        QStringLiteral("baseline    64x128     96     212  11.25"),
        QStringLiteral("128x256     128x256   168      40  17.90"),
        QStringLiteral("dcp4-split  64x128    128       0  22.40"),
        QStringLiteral("128-reg     128x128   128       0  29.60"),
    };
    for (const QString &r : rows) {
        p.drawText(M, y, r);
        y += 30;
    }
    y += 40;
    p.setFont(body);
    const QStringList para2 = {
        QStringLiteral("Acceptance: greedy logprob parity on 24 prompts, max abs diff"),
        QStringLiteral("below 2e-3 against the BF16 reference decode."),
    };
    for (const QString &l : para2) {
        p.drawText(M, y, l);
        y += 34;
    }
    p.end();
}

void writeWav(const QString &path)
{
    const int rate = 24000;
    const int n = rate * 2;
    QByteArray pcm(n * 2, '\0');
    auto *s = reinterpret_cast<qint16 *>(pcm.data());
    for (int i = 0; i < n; ++i) {
        const double t = double(i) / rate;
        const double env = 0.15 + 0.85 * std::pow(std::sin(M_PI * t * 3.0), 2.0) * std::exp(-0.6 * t);
        const double v = env * (0.6 * std::sin(2 * M_PI * 220 * t) + 0.3 * std::sin(2 * M_PI * 440 * t * (1 + 0.1 * t)));
        s[i] = qToLittleEndian<qint16>(qint16(std::clamp(v, -1.0, 1.0) * 30000));
    }
    QByteArray h;
    auto u32 = [&](quint32 v) { char b[4]; qToLittleEndian(v, b); h.append(b, 4); };
    auto u16 = [&](quint16 v) { char b[2]; qToLittleEndian(v, b); h.append(b, 2); };
    h.append("RIFF");
    u32(36 + pcm.size());
    h.append("WAVEfmt ");
    u32(16);
    u16(1);
    u16(1);
    u32(rate);
    u32(rate * 2);
    u16(2);
    u16(16);
    h.append("data");
    u32(pcm.size());
    QFile f(path);
    if (f.open(QIODevice::WriteOnly)) {
        f.write(h);
        f.write(pcm);
    }
}

void writeSvg(const QString &path)
{
    QString svg = QStringLiteral(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1200\" height=\"640\" viewBox=\"0 0 1200 640\">"
        "<rect width=\"1200\" height=\"640\" fill=\"#0d0c0b\"/>");
    for (int i = 0; i < 40; ++i) {
        const double h = 60 + 420 * std::exp(-std::pow((i - 12) / 5.0, 2.0)) + 30 * std::sin(i);
        const bool hot = i >= 10 && i <= 14;
        svg += QStringLiteral("<rect x=\"%1\" y=\"%2\" width=\"22\" height=\"%3\" rx=\"4\" fill=\"%4\"/>")
                   .arg(60 + i * 27)
                   .arg(560 - h, 0, 'f', 1)
                   .arg(h, 0, 'f', 1)
                   .arg(hot ? QStringLiteral("#e09f3e") : QStringLiteral("#b5653b"));
    }
    svg += QStringLiteral("<rect x=\"60\" y=\"572\" width=\"1080\" height=\"3\" fill=\"#3a332c\"/></svg>");
    QFile f(path);
    if (f.open(QIODevice::WriteOnly))
        f.write(svg.toUtf8());
}

Turn user(const QString &t) { return Turn{QStringLiteral("user"), t, {}, {}, {}, {}, {}, false}; }
Turn assistant(const QString &t, const QString &thinking = {}) { return Turn{QStringLiteral("assistant"), t, thinking, {}, {}, {}, {}, false}; }
Turn withTools(Turn t, const QStringList &tools, const QStringList &files)
{
    t.tools = tools;
    t.files = files;
    return t;
}

}  // namespace

void ensureFixtureFiles()
{
    if (!QFileInfo::exists(specPdfPath()))
        writeSpecPdf(specPdfPath());
    if (!QFileInfo::exists(wavPath()))
        writeWav(wavPath());
    if (!QFileInfo::exists(svgPath()))
        writeSvg(svgPath());
}

QStringList fixtureFolders()
{
    return {QStringLiteral("~/dev/fp4-decode"), QStringLiteral("~/dev/cluster-serving"),
            QStringLiteral("~/dev/kernel-search"), QStringLiteral("~/dev/phonon"), QStringLiteral("~/dev/lapis"),
            QStringLiteral("~/dev/yoctogpt")};
}

QList<Turn> kickoffTurns()
{
    const QString diff = QStringLiteral(
        "```diff\n"
        "--- a/csrc/decode_fp4.cu\n"
        "+++ b/csrc/decode_fp4.cu\n"
        "@@ -118,13 +118,15 @@ __global__ void decode_fp4_kv(\n"
        "   const int lane = threadIdx.x & 31;\n"
        "-  float acc[64];\n"
        "-  #pragma unroll 4\n"
        "-  for (int t = 0; t < TILE_K; t += 64) {\n"
        "+  float acc[128];  // 128-reg tile: dequant stays in registers\n"
        "+  const float scale_reg = scales[blockIdx.y];\n"
        "+  #pragma unroll\n"
        "+  for (int t = 0; t < TILE_K; t += 128) {\n"
        "     uint32_t packed = kv_fp4[base + t / 8];\n"
        "-    half2 v = dequant_fp4(packed, scale_smem[t / 32]);\n"
        "-    __syncthreads();\n"
        "+    half2 v = dequant_fp4(packed, scale_reg);\n"
        "+    acc[t / 2] = __fmaf_rn(__low2float(v), q_reg[t & 127], acc[t / 2]);\n"
        "     acc[t / 2 + 1] += __high2float(v) * q_reg[(t + 1) & 127];\n"
        "   }\n"
        "-  store_partial(out, acc, 64);\n"
        "+  store_partial(out, acc, 128);\n"
        "   if (lane == 0) atomicAdd(&done[blockIdx.x], 1);\n"
        "```\n");
    QString csv = QStringLiteral("```csv\narm,seq,tile,regs,spills,tok/s,speedup\n");
    const char *arms[4] = {"baseline", "128x256", "dcp4-split", "128-reg"};
    const char *tiles[4] = {"64x128", "128x256", "64x128", "128x128"};
    const int regs[4] = {96, 168, 128, 128};
    const int spills[4] = {212, 40, 0, 0};
    const double base[4] = {11.25, 17.9, 22.4, 29.6};
    const int seqs[5] = {4096, 8192, 16384, 32768, 65536};
    for (int s = 0; s < 5; ++s)
        for (int a = 0; a < 4; ++a) {
            const double scale = 1.0 + 0.18 * (2 - s);
            const double v = base[a] * scale;
            csv += QStringLiteral("%1,%2,%3,%4,%5,%6,%7x\n")
                       .arg(QLatin1String(arms[a]))
                       .arg(seqs[s])
                       .arg(QLatin1String(tiles[a]))
                       .arg(regs[a])
                       .arg(spills[a])
                       .arg(v, 0, 'f', 1)
                       .arg(base[a] / base[0], 0, 'f', 2);
        }
    csv += QStringLiteral("```\n");

    const QString text = QStringLiteral(
        "Ran 4 arms with DCP=4 and MTP=3 held fixed. 128-reg won at 29.6 tok/s, 2.63x over baseline.\n\n"
        "| arm | tok/s | decode speedup |\n"
        "|---|---|---|\n"
        "| baseline | 11.25 | 1.00x |\n"
        "| 128x256 | 17.9 | 1.59x |\n"
        "| dcp4-split | 22.4 | 1.99x |\n"
        "| 128-reg | 29.6 | 2.63x |\n\n"
        "The win comes from keeping the FP4 dequant entirely in registers: the 128-register tile "
        "removes the shared-memory scale lookup and the barrier that followed it, and the larger "
        "unroll hides the global load latency of the packed KV words. The 128x256 arm spilled 40 "
        "registers per thread; dcp4-split removed spills but kept the barrier.\n\n")
        + diff + QStringLiteral("\nPer-sequence sweep:\n") + csv
        + QStringLiteral("\nSpec with the parity gate: ") + specPdfPath() + QLatin1Char('\n');
    return {user(QStringLiteral("profile fp4 kv-cache decode on sm120")),
            assistant(text, QStringLiteral("baseline spills 212 regs; try a 128-register tile"))};
}

QList<FixtureProject> fixtureProjects()
{
    QList<FixtureProject> out;
    out << FixtureProject{
        QStringLiteral("cluster-serving"), QStringLiteral("~/dev/cluster-serving"), QStringLiteral("working"),
        {user(QStringLiteral("serve the 65B with DCP=4 on 4 cards")),
         assistant(QStringLiteral("Baseline layout is TP4 across the four cards; the KV split comes next.")),
         user(QStringLiteral("compare against TP4 and add MTP")),
         withTools(assistant(QStringLiteral(
                       "DCP=4 KV split lands 226 tok/s at c1, 1.35x over TP4.\n\n"
                       "| layout | tok/s |\n|---|---|\n| TP4 | 167 |\n| DCP=2 | 198 |\n| DCP=4 | 226 |\n| DCP=4 MTP | 241 |\n")),
                   {QStringLiteral("Bash: ./serve.sh --dcp 4 --mtp && ./bench_c1.py"), QStringLiteral("Edit ~/dev/cluster-serving/serve.sh")},
                   {QStringLiteral("~/dev/cluster-serving/serve.sh"), QStringLiteral("~/dev/cluster-serving/bench_c1.py")})}};
    out << FixtureProject{
        QStringLiteral("kernel-search"), QStringLiteral("~/dev/kernel-search"), QStringLiteral("finished"),
        {user(QStringLiteral("search fused rmsnorm kernels")),
         assistant(QStringLiteral(
             "Best fused kernel is 4.10x over eager.\n\n"
             "| kernel | speedup |\n|---|---|\n| eager | 1.00x |\n| triton | 2.35x |\n| cuda-v1 | 3.20x |\n| cuda-v4 | 4.10x |\n"))}};
    out << FixtureProject{
        QStringLiteral("phonon"), QStringLiteral("~/dev/phonon"), QStringLiteral("idle"),
        {user(QStringLiteral("fix the fn key toggle latency")),
         assistant(QStringLiteral(
             "Toggle path now 38 ms.\n\n| build | latency ms |\n|---|---|\n| v0.9 | 142 |\n| v1.0 | 81 |\n| v1.1 | 38 |\n\n"
             "Recorded click: ") + wavPath() + QLatin1Char('\n'))}};
    out << FixtureProject{
        QStringLiteral("lapis"), QStringLiteral("~/dev/lapis"), QStringLiteral("working"),
        {user(QStringLiteral("frame pacing at 240 Hz")),
         assistant(QStringLiteral("Frame p99 is 4.2 ms at 240 Hz, 2.1x better. Histogram: ") + svgPath()
                   + QStringLiteral("\n\n| build | p99 ms |\n|---|---|\n| main | 8.8 |\n| pacing | 4.2 |\n"))}};
    QString loss = QStringLiteral("| step | loss |\n|---|---|\n");
    for (int i = 0; i <= 20; ++i)
        loss += QStringLiteral("| %1 | %2 |\n").arg(i * 500).arg(2.41 + 3.2 * std::exp(-i / 4.0) + 0.04 * std::sin(i * 1.7), 0, 'f', 3);
    out << FixtureProject{QStringLiteral("yoctogpt"), QStringLiteral("~/dev/yoctogpt"), QStringLiteral("finished"),
                          {user(QStringLiteral("train the 12M baseline")),
                           assistant(QStringLiteral("Loss settles at 2.41, 61 tok/s on CPU.\n\n") + loss)}};
    return out;
}

QByteArray claudeSampleJsonl()
{
    return QByteArray(
        R"({"type":"summary","summary":"x"})" "\n"
        R"({"type":"user","message":{"role":"user","content":"profile fp4 decode with DCP=4"}})" "\n"
        R"({"type":"user","isMeta":true,"message":{"role":"user","content":"meta noise"}})" "\n"
        R"({"type":"assistant","message":{"role":"assistant","content":[{"type":"thinking","thinking":"spills dominate","signature":"s"},{"type":"text","text":"Won at 29.6 tok/s, 2.63x.\n\n| arm | tok/s |\n|---|---|\n| base | 11.25 |\n| 128-reg | 29.6 |\n"}]}})" "\n"
        R"({"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Edit","input":{"file_path":"/tmp/k.cu","old_string":"float acc[64];","new_string":"float acc[128];"}},{"type":"tool_use","name":"Bash","input":{"command":"nvcc -O3 /tmp/k.cu && ./a.out","description":"build"}}]}})" "\n"
        R"({"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"ok"}]}})" "\n"
        R"({"type":"user","message":{"role":"user","content":"<system-reminder>skip</system-reminder>"}})" "\n");
}

QByteArray codexSampleJsonl()
{
    return QByteArray(
        R"({"timestamp":"t","type":"session_meta","payload":{"id":"x"}})" "\n"
        R"({"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"<environment_context>x</environment_context>"}]}})" "\n"
        R"({"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"tune the decode tile"}]}})" "\n"
        R"({"type":"response_item","payload":{"type":"reasoning","summary":[{"type":"summary_text","text":"compare tiles"}]}})" "\n"
        R"({"type":"response_item","payload":{"type":"function_call","name":"shell","arguments":"{\"command\":[\"bash\",\"-lc\",\"cargo bench --bench tile\"],\"workdir\":\"/tmp/tile\"}"}})" "\n"
        R"({"type":"response_item","payload":{"type":"custom_tool_call","name":"apply_patch","input":"*** Begin Patch\n*** Update File: src/tile.rs\n@@ fn tile()\n-    64\n+    128\n*** End Patch"}})" "\n"
        R"({"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"```csv\ntile,ms\n64,3.1\n128,1.7\n```\nPlot: /nonexistent/plot.png"}]}})" "\n"
        R"({"type":"event_msg","payload":{"type":"token_count"}})" "\n");
}

QByteArray agySampleJsonl()
{
    return QByteArray(
        R"({"step_index":0,"type":"USER_INPUT","content":"<USER_REQUEST>\nwhy did 128-reg win?\n</USER_REQUEST>\n<ADDITIONAL_METADATA>m</ADDITIONAL_METADATA>"})" "\n"
        R"({"step_index":1,"type":"GENERIC","content":"Created file."})" "\n"
        R"({"step_index":2,"type":"PLANNER_RESPONSE","content":"Registers hold the dequant; 1.8x fewer stalls."})" "\n");
}

}  // namespace studio
