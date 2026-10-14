"""Minimal scouting NanoAOD for the MC ID workflow (not the flat regression tree)."""

import importlib
import json

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing("analysis")
options.register("job", "job.json", VarParsing.multiplicity.singleton,
                 VarParsing.varType.string, "Resolved job specification")
options.parseArguments()
with open(options.job) as stream:
    job = json.load(stream)
physics = job["physics"]
modifiers = [getattr(importlib.import_module(module), name)
             for module, name in (entry.rsplit(":", 1) for entry in physics["modifiers"])]
process = cms.Process("SCOUTNANO", *modifiers)
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = physics["report_every"]
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(job["max_events"]))
process.options = cms.untracked.PSet(
    numberOfThreads=cms.untracked.uint32(job["threads"]),
    numberOfStreams=cms.untracked.uint32(0),
    wantSummary=cms.untracked.bool(True),
)
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(job["input_files"]),
                            skipBadFiles=cms.untracked.bool(False))
process.load("PhysicsTools.NanoAOD.run3scouting_cff")
if not hasattr(process, "scoutingElectronRegressVarTable"):
    scouting = importlib.import_module("PhysicsTools.NanoAOD.run3scouting_cff")
    raise RuntimeError(
        f"Loaded {scouting.__file__}, but scoutingElectronRegressVarTable is missing. "
        "Install the custom PhysicsTools/NanoAOD and PhysicsTools/Scouting packages "
        "in cmssw.base on a named Git branch, run scram b, and check the imported "
        "module path. See the CMSSW setup in README.md."
    )
for name, values in physics["best_track"].items():
    if not hasattr(process.scoutingElectronBestTrack, name):
        raise ValueError("Unknown best-track parameter: " + name)
    setattr(process.scoutingElectronBestTrack, name, cms.vdouble(values))
process.nanoTask = cms.Task(
    process.scoutingElectronBestTrack, process.scoutingElectronRegressVarTable,
    process.scoutingElectronTable, process.scoutingPhotonTable,
    process.scoutingPrimaryVertexTable, process.scoutingRhoTable,
)
if job["is_mc"]:
    process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
    process.load("PhysicsTools.NanoAOD.genparticles_cff")
    process.load("PhysicsTools.NanoAOD.globals_cff")
    process.nanoTask.add(process.genParticleTask, process.genParticleTablesTask, process.puTable)
process.nanoPath = cms.Path(process.nanoTask)
process.output = cms.OutputModule(
    "NanoAODOutputModule",
    fileName=cms.untracked.string("scouting_nano.root"),
    outputCommands=cms.untracked.vstring("drop *", "keep nanoaodFlatTable_*_*_*"),
    compressionAlgorithm=cms.untracked.string("ZSTD"),
    compressionLevel=cms.untracked.int32(9),
    saveProvenance=cms.untracked.bool(False),
)
process.outputPath = cms.EndPath(process.output)
process.schedule = cms.Schedule(process.nanoPath, process.outputPath)
