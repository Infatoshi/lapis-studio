// Fixture data for the mock backend (--test, --record-demo). Fixture projects are
// synthetic transcripts fed through the real extraction pipeline.
#pragma once

#include "transcript.hpp"

#include <QByteArray>
#include <QList>
#include <QString>

namespace studio {

struct FixtureProject {
    QString key;
    QString dir;
    QString status;  // working | finished | idle
    QList<Turn> turns;
};

QString fixtureDir();          // <exe dir>/fixtures, created on demand
void ensureFixtureFiles();     // spec PDF, waveform WAV, SVG figure
// <fixtureDir>/slowmix: a small clips dir (songs, junk takes, a baked export) plus a .plays.tsv
// whose timestamps are relative to now. Returns the dir.
QString ensureSlowmixFixture();
QList<FixtureProject> fixtureProjects();
QList<Turn> kickoffTurns();    // the fp4 decode session the demo starts
QStringList fixtureFolders();

// Raw JSONL samples in each harness's real on-disk format, for parser tests.
QByteArray claudeSampleJsonl();
QByteArray codexSampleJsonl();
QByteArray agySampleJsonl();

}  // namespace studio
