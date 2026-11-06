"""Readable, explicitly approximate Gaussian-trajectory diagnostics.

This renderer never infers exact TDPES or a corrected quantum momentum from
trajectory density. Joint maps use an explicitly labelled visualization KDE;
a/b are the actual trajectory values (scatter), not an interpolated EF field.
Color limits are fixed over the saved interval. Movie PNGs are temporary to
avoid multiplying result storage; representative high-resolution PNGs persist.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np


STYLE = {"font.size": 13, "axes.titlesize": 17, "axes.labelsize": 15,
         "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10,
         "figure.titlesize": 21, "axes.spines.top": False,
         "axes.spines.right": False, "lines.linewidth": 2.0}
COLORS = ["#202b55", "#d45920", "#15967d", "#984ba2", "#9c7733", "#4b8fb8"]
FRAME_KEYS = ("t_fs", "R", "P", "eta", "q", "quadrature", "rho", "a", "b",
              "delta_b", "alpha", "heavy_eval_R", "heavy_density", "Q_raw", "z",
              "projection_residual", "pnc_error", "MLS_condition", "population",
              "photon_number", "energy", "kinetic_R", "bo_population")


def _load_frame(path):
    # Deliberately do not load G: maps and diagnostics already have cached fields.
    with np.load(path, allow_pickle=False) as z:
        d = {key: z[key].copy() for key in FRAME_KEYS if key in z}
    for key in ("R", "eta", "q", "quadrature", "rho", "t_fs"):
        if key not in d:
            raise ValueError("%s is missing required cached field %s" % (path, key))
    d["t_fs"] = float(d["t_fs"])
    return d


def _weighted(frame, key):
    return np.einsum("i,i...->...", frame["eta"], frame[key])


def _nice_limit(values, *, symmetric=False, pad=0.08):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        bound = max(float(np.quantile(abs(a), 0.995)), 1e-12)
        ticks = MaxNLocator(nbins=5).tick_values(-bound, bound)
        return (-max(abs(ticks[0]), abs(ticks[-1])), max(abs(ticks[0]), abs(ticks[-1])))
    lo, hi = float(a.min()), float(a.max())
    span = max(hi-lo, 0.05*max(abs(lo), abs(hi)), 1e-6)
    return lo-pad*span, hi+pad*span


def _axis(ax, xlabel, ylabel, title):
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_powerlimits((-3, 4))
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(alpha=0.16)


def _missing(ax, text):
    ax.text(0.5, 0.5, text + "\nnot available", ha="center", va="center",
            transform=ax.transAxes, color="#555555")


def _legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best", ncol=2 if len(handles)>3 else 1,
                  framealpha=0.85, borderpad=0.35)


def prepare(root, *, visual_h_R=None, support_relative=1e-5):
    """Small streaming prepass; return file list, traces and stable plotting limits."""
    root = Path(root)
    files = sorted((root / "frames").glob("frame_*.npz"))
    if not files:
        files = sorted(root.glob("frame_*.npz"))
    if not files:
        raise FileNotFoundError("No frame_*.npz in " + str(root))
    if not 0 < support_relative < 1:
        raise ValueError("support_relative must lie between zero and one")
    first = _load_frame(files[0])
    q = first["q"]
    mean_R = first["eta"] @ first["R"]
    std_R = np.sqrt(first["eta"] @ (first["R"]-mean_R)**2)
    if visual_h_R is None:
        visual_h_R = 0.5*float(std_R) if std_R > 0 else None
    elif not np.isfinite(visual_h_R) or visual_h_R <= 0:
        raise ValueError("visual_h_R must be finite and positive")
    traces = {key: [] for key in ("times_fs", "energy", "kinetic_R", "photon_number",
                                  "population", "projection_residual", "pnc_error",
                                  "mls_condition", "rho_q", "centers", "widths", "weights")}
    field_samples = {key: [] for key in ("a", "b", "delta_b")}
    R_values, rho_peak, conditional_peak = [], 0.0, 0.0
    heavy_peak, momentum_samples = 0.0, []
    occupied_q = np.zeros(q.shape,dtype=bool)
    config_file = root / "config.json"
    config = json.loads(config_file.read_text()) if config_file.exists() else {}
    metadata_file=root/"metadata.json"
    metadata=json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    model_metadata=metadata.get("model",{})
    model_metadata=model_metadata if isinstance(model_metadata,dict) else {}
    basis=model_metadata.get("basis_convention",model_metadata.get("basis","declared electronic/channel basis"))
    population_basis="local BO states" if "bo_population" in first else basis
    coordinates = config.get("gaussian_coordinates", "transformed")
    for path in files:
        f = _load_frame(path)
        if f["q"].shape != q.shape or not np.allclose(f["q"], q, rtol=0, atol=1e-12):
            raise ValueError("movie requires the common q grid declared in the method")
        traces["times_fs"].append(f["t_fs"])
        for key in ("energy", "kinetic_R", "photon_number", "population"):
            source="bo_population" if key=="population" and population_basis=="local BO states" else key
            traces[key].append(_weighted(f,source) if source in f else np.nan)
        for key, source in (("projection_residual", "projection_residual"),
                            ("pnc_error", "pnc_error"), ("mls_condition", "MLS_condition")):
            traces[key].append(float(np.max(f[source])) if source in f else np.nan)
        rho_q = _weighted(f, "rho")
        traces["rho_q"].append(rho_q)
        rho_peak = max(rho_peak, float(np.max(rho_q)))
        conditional_peak = max(conditional_peak, float(np.max(f["rho"])))
        occupied_q |= rho_q > support_relative*np.max(rho_q)
        if "heavy_density" in f:
            heavy_peak=max(heavy_peak,float(np.max(f["heavy_density"])))
        for key in ("alpha","Q_raw"):
            if key in f:
                momentum_samples.extend(np.asarray(f[key]).ravel().tolist())
        R_values.extend(f["R"].tolist())
        for key in field_samples:
            if key in f:
                keep = f["rho"] > support_relative*np.max(f["rho"], axis=1, keepdims=True)
                values = np.asarray(f[key])[keep]
                field_samples[key].extend(values[::max(1,len(values)//1000)].tolist())
        if "z" in f and f["z"].ndim==2 and f["z"].shape[1]>=2:
            from .gaussian import unpack_parameters
            K = (f["z"].shape[-1]+1)//3
            pars = [unpack_parameters(z,K,coordinates=coordinates) for z in f["z"]]
            for key, j in (("weights",0),("centers",1),("widths",2)):
                traces[key].append(np.einsum("i,ij->j",f["eta"],np.array([p[j] for p in pars])))
        else:
            for key in ("weights","centers","widths"):
                traces[key].append(np.nan)
    traces = {key: np.asarray(value) for key,value in traces.items()}
    limits = {key: _nice_limit(values, symmetric=True) for key,values in field_samples.items()}
    # b and b-alpha have the same physical units and share a comparison range.
    bound = max(limits["b"][1],limits["delta_b"][1])
    limits["b"] = limits["delta_b"] = (-bound,bound)
    qlims=_nice_limit(q[occupied_q],pad=.08)
    limits.update({"R": _nice_limit(R_values, pad=.15),
                   "q": (max(float(q[0]),qlims[0]),min(float(q[-1]),qlims[1])),
                   "density_q": (0.,1.12*rho_peak), "density_conditional": (0.,1.12*conditional_peak),
                   "density_heavy": (0.,max(1e-12,1.12*heavy_peak)),
                   "outer_momentum": _nice_limit(momentum_samples,symmetric=True)})
    data={"root":root,"files":files,"first":first,"q":q,"traces":traces,
            "limits":limits,"visual_h_R":visual_h_R,"support_relative":support_relative,
            "config":config,"population_basis":population_basis,
            "run_label":("Uncompressed MTE reference" if config.get("compression") in ("native","none") else
                         "MTE + Gaussian bridge projection")}
    if config.get("method") in ("ctmqc", "tsh"):
        method = "CTMQC equations" if config["method"] == "ctmqc" else "FSSH"
        data["run_label"] = method + (" + Gaussian bridge projection (experimental)"
            if config.get("compression") == "gaussian" else " full-internal uncompressed pilot")
    # Stable joint-density color scale, evaluated on the same visualization grid.
    joint_peak=0.0
    if visual_h_R is not None:
        for path in files:
            _,joint=_joint_visualization(_load_frame(path),data)
            joint_peak=max(joint_peak,float(np.max(joint)))
    limits["joint_density"]=(0.,max(joint_peak,1e-12))
    return data


def _parameter_panel(ax, data):
    t = data["traces"]["times_fs"]
    centers, widths = data["traces"]["centers"], data["traces"]["widths"]
    if centers.ndim != 2:
        if data["config"].get("compression") in ("native","none"):
            ax.text(.5,.5,"Uncompressed reference\nNo Gaussian parameters",ha="center",va="center",transform=ax.transAxes)
        else:
            _missing(ax,"Gaussian parameters")
    else:
        for j in range(min(centers.shape[1],4)):
            ax.plot(t,centers[:,j],color=COLORS[j],label="$Q_%d$"%(j+1))
            ax.plot(t,widths[:,j],color=COLORS[j],ls="--",label="$s_%d$"%(j+1))
        if centers.shape[1]>4:
            ax.text(.02,.02,"First four components",transform=ax.transAxes)
        _legend(ax)
    _axis(ax,"t (fs)","q-coordinate units","Gaussian centers / widths (ensemble means)")


def _population_panel(ax, data):
    t,p = data["traces"]["times_fs"],data["traces"]["population"]
    if p.ndim != 2:
        _missing(ax,"Electronic/channel populations")
    elif "electronic Dirichlet grid" in data["population_basis"]:
        if len(t)>1:
            m=ax.pcolormesh(t,np.arange(p.shape[1]),p.T,shading="auto",cmap="magma",vmin=0,vmax=float(p.max()))
            ax.figure.colorbar(m,ax=ax,label="Grid-bin probability")
        else:
            ax.plot(np.arange(p.shape[1]),p[0])
        _axis(ax,"t (fs)" if len(t)>1 else "Electronic grid index",
              "Electronic grid index" if len(t)>1 else "Grid-bin probability",
              "Electronic primitive-grid probabilities (not BO)")
        return
    else:
        for k in range(min(p.shape[1],6)):
            label=("BO %d"%k if data["population_basis"]=="local BO states" else
                   "diabatic %d"%k if "diabatic" in data["population_basis"] else "channel %d"%k)
            ax.plot(t,p[:,k],color=COLORS[k],label=label)
        if p.shape[1]>6:
            ax.plot(t,p[:,6:].sum(axis=1),color="0.5",ls="--",label="remaining channels")
        _legend(ax)
    ax.set_ylim(-.03,1.03)
    title=("Local BO populations" if data["population_basis"]=="local BO states" else
           "Diabatic electronic populations" if "diabatic" in data["population_basis"] else
           "Declared electronic/channel populations")
    _axis(ax,"t (fs)","Population",title)


def _energy_panel(ax, data):
    tr=data["traces"]
    for key,label,color in (("energy","Total (declared backend)",COLORS[0]),
                            ("kinetic_R","Heavy kinetic",COLORS[1])):
        if np.any(np.isfinite(tr[key])):
            ax.plot(tr["times_fs"],tr[key],label=label,color=color)
    _axis(ax,"t (fs)","Energy (Ha)","Energy diagnostics; no conservation assumed")
    _legend(ax)


def plot_summary(data):
    """Nine-panel overview without claiming exact scalar fields or corrected Q."""
    tr,t=data["traces"],data["traces"]["times_fs"]
    fig,axes=plt.subplots(3,3,figsize=(21,16),layout="constrained")
    _population_panel(axes[0,0],data)
    ax=axes[0,1]
    if np.any(np.isfinite(tr["photon_number"])):
        ax.plot(t,tr["photon_number"],color=COLORS[1])
        _axis(ax,"t (fs)",r"$\langle n_{\rm ph}\rangle$","Photon number from complex internal state")
    else:
        means=np.einsum("tq,q,q->t",tr["rho_q"],data["q"],data["first"]["quadrature"])
        ax.plot(t,means,color=COLORS[1])
        _axis(ax,"t (fs)","Mean q (a.u.)","Bridge coordinate expectation")
    _energy_panel(axes[0,2],data)
    for ax,key,label,title in ((axes[1,0],"projection_residual",r"$\|\dot G_{\rm actual}-\dot G_{\rm EOM}\|_2$ (a.u.)","Gaussian compression rate residual"),
                              (axes[1,1],"pnc_error","Max raw norm / PNC error","Raw normalization defect (no correction)"),
                              (axes[1,2],"mls_condition","Condition number","Local R reconstruction conditioning")):
        values=tr[key]
        valid=np.isfinite(values)&(values>0)
        if np.any(valid):
            ax.semilogy(t[valid],values[valid],color=COLORS[0])
        elif np.all(values==0):
            ax.plot(t,values,color=COLORS[0]); ax.text(.04,.7,"Exactly zero in saved data",transform=ax.transAxes)
        else:
            _missing(ax,title)
        _axis(ax,"t (fs)",label,title)
        if ax.get_yscale()=="log":
            if key == "mls_condition":
                ax.set_ylim(1., max(10., 1.2*np.max(values[valid])))
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
            ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
    ax=axes[2,0]
    if len(t)>1:
        m=ax.pcolormesh(t,data["q"],tr["rho_q"].T,shading="auto",cmap="magma",vmin=0,
                        vmax=data["limits"]["density_q"][1])
        fig.colorbar(m,ax=ax,label="Density")
        _axis(ax,"t (fs)","q (a.u.)","Bridge marginal (trajectory-weighted)")
    else:
        ax.plot(data["q"],tr["rho_q"][0]); _axis(ax,"q (a.u.)","Density","Bridge marginal")
    _parameter_panel(axes[2,1],data)
    ax=axes[2,2]
    if tr["weights"].ndim==2:
        for j in range(min(tr["weights"].shape[1],6)):
            ax.plot(t,tr["weights"][:,j],label="$w_%d$"%(j+1),color=COLORS[j])
        ax.set_ylim(-.03,1.03); _legend(ax)
    else:
        if data["config"].get("compression") in ("native","none"):
            ax.text(.5,.5,"Uncompressed reference\nNo Gaussian mixture weights",ha="center",va="center",transform=ax.transAxes)
        else:
            _missing(ax,"Gaussian mixture weights")
    _axis(ax,"t (fs)","Mixture weight","Gaussian weights (not electronic populations)")
    fig.suptitle(data["run_label"]+" — approximate dynamics",fontsize=23)
    return fig


def _joint_visualization(frame,data):
    h=data["visual_h_R"]
    if h is None:
        return None,None
    Rgrid=np.linspace(*data["limits"]["R"],128)
    kernels=np.exp(-.5*((Rgrid[:,None]-frame["R"])/h)**2)/(np.sqrt(2*np.pi)*h)
    joint=kernels@(frame["eta"][:,None]*frame["rho"])
    return Rgrid,joint


def _contours(ax,q,Rgrid,joint):
    if joint is None or joint.max()<=0:
        return
    levels=joint.max()*np.geomspace(1e-4,.95,12)
    ax.contour(q,Rgrid,joint,levels=levels,colors="black",linewidths=.75,alpha=.85)


def plot_snapshot(frame,data):
    """Twelve panels, fixed movie limits, actual field samples plus KDE contours."""
    fig,axes=plt.subplots(4,3,figsize=(23,23),layout="constrained")
    q=frame["q"]; t=frame["t_fs"]
    index=int(np.argmin(abs(frame["R"]-np.median(frame["R"]))))
    Rsel=float(frame["R"][index]); rho_q=_weighted(frame,"rho")
    ax=axes[0,0]
    ax.fill_between(q,frame["rho"][index],color=".75",alpha=.8)
    ax.plot(q,frame["rho"][index],color=COLORS[0])
    _axis(ax,"q (a.u.)",r"$|\Lambda_R|^2$","Conditional bridge at R = %.3g a.u."%Rsel)
    ax.set_ylim(*data["limits"]["density_conditional"])
    ax.set_xlim(*data["limits"]["q"])
    ax=axes[0,1]
    ax.fill_between(q,rho_q,color=".75",alpha=.8); ax.plot(q,rho_q,color=COLORS[0])
    _axis(ax,"q (a.u.)","Density","Trajectory-weighted q marginal")
    ax.set_ylim(*data["limits"]["density_q"])
    ax.set_xlim(*data["limits"]["q"])
    ax=axes[0,2]
    if "heavy_eval_R" in frame and "heavy_density" in frame:
        ax.fill_between(frame["heavy_eval_R"],frame["heavy_density"],color=".75",alpha=.8)
        ax.plot(frame["heavy_eval_R"],frame["heavy_density"],color=COLORS[0])
    else:
        _missing(ax,"Heavy density reconstruction")
    ax.axvline(Rsel,color=".35",ls="--",lw=1.3)
    _axis(ax,"R (a.u.)",r"$|\chi|^2_{\rm rec}$","Heavy density: declared estimator")
    ax.set_xlim(*data["limits"]["R"])
    ax.set_ylim(*data["limits"]["density_heavy"])
    Rgrid,joint=_joint_visualization(frame,data)
    ax=axes[1,0]
    if joint is not None:
        m=ax.pcolormesh(q,Rgrid,joint,shading="auto",cmap="magma",
                       norm=Normalize(*data["limits"]["joint_density"]))
        fig.colorbar(m,ax=ax,label="Joint density")
        _contours(ax,q,Rgrid,joint)
        ax.text(.02,.98,"Visualization KDE: h_R = %.3g"%data["visual_h_R"],transform=ax.transAxes,
                va="top",color="white",bbox={"facecolor":"black","alpha":.65,"pad":3})
    else:
        _missing(ax,"Joint KDE (set visual_h_R for fixed R)")
    _axis(ax,"q (a.u.)","R (a.u.)","Joint density: KDE in R, quantum q")
    for ax,key,label in ((axes[1,1],"a",r"$a$"),(axes[1,2],"b",r"$b$"),
                          (axes[2,0],"delta_b",r"$b-\alpha$")):
        if key not in frame:
            _missing(ax,label)
        else:
            mask=frame["rho"]>data["support_relative"]*frame["rho"].max(axis=1,keepdims=True)
            qq=np.broadcast_to(q,frame["rho"].shape)
            RR=np.broadcast_to(frame["R"][:,None],frame["rho"].shape)
            m=ax.scatter(qq[mask],RR[mask],c=frame[key][mask],s=10,cmap="RdBu_r",
                         norm=Normalize(*data["limits"][key]),rasterized=True,marker="s",linewidths=0)
            fig.colorbar(m,ax=ax,label="Momentum (a.u.)")
            _contours(ax,q,Rgrid,joint)
            ax.axhline(Rsel,color=".25",ls="--",lw=1.1)
        _axis(ax,"q (a.u.)","R (a.u.)",label+" — trajectory samples")
    for ax in (axes[1,0],axes[1,1],axes[1,2],axes[2,0]):
        ax.set_xlim(*data["limits"]["q"]); ax.set_ylim(*data["limits"]["R"])
    ax=axes[2,1]
    for key,label,color in (("alpha",r"$\alpha$",COLORS[0]),("Q_raw",r"$\mathcal{P}_{R,\rm density}$",COLORS[1])):
        if key in frame:
            ax.scatter(frame["R"],frame[key],label=label,color=color,s=20)
    ax.axhline(0,color=".4",ls="--",lw=1); ax.set_xlim(*data["limits"]["R"])
    ax.set_ylim(*data["limits"]["outer_momentum"])
    _axis(ax,"R (a.u.)","Momentum (a.u.)","Connection / raw quantum momentum (distinct)")
    _legend(ax)
    _population_panel(axes[2,2],data); axes[2,2].axvline(t,color=".4",ls="--",lw=1)
    _energy_panel(axes[3,0],data); axes[3,0].axvline(t,color=".4",ls="--",lw=1)
    _parameter_panel(axes[3,1],data); axes[3,1].axvline(t,color=".4",ls="--",lw=1)
    ax=axes[3,2]
    tr=data["traces"]
    values=tr["projection_residual"]
    valid=np.isfinite(values)
    if np.any(valid):
        ax.plot(tr["times_fs"][valid],values[valid],color=COLORS[0])
    else:
        _missing(ax,"Amplitude projection residual")
    ax.axvline(t,color=".4",ls="--",lw=1)
    _axis(ax,"t (fs)",r"$\|\dot G_{\rm actual}-\dot G_{\rm EOM}\|_2$ (a.u.)","Compression error (not QD discrepancy)")
    ax.text(.03,.95,"Raw norm/PNC: %.2e\nMax MLS condition: %.2e"%
            (float(np.max(frame.get("pnc_error",np.nan))),float(np.max(frame.get("MLS_condition",np.nan)))),
            transform=ax.transAxes,va="top",bbox={"facecolor":"white","alpha":.8,"edgecolor":".8"})
    for j,ax in enumerate(axes.flat):
        ax.text(-.11,1.08,chr(65+j),transform=ax.transAxes,fontweight="bold",fontsize=18)
    fig.suptitle(data["run_label"]+" | t = %.4f fs\n"
                 "Approximate fields; black contours = visualization joint-density KDE"%t,fontsize=23)
    return fig


def render(root, movies=False, *, visual_h_R=None, support_relative=1e-5,
           dpi=130, movie_dpi=70, fps=8, snapshot_indices=None):
    """Generate summary, snapshots and optional MP4; missing encoder is nonfatal."""
    if dpi<=0 or movie_dpi<=0 or fps<=0:
        raise ValueError("dpi, movie_dpi and fps must be positive")
    data=prepare(root,visual_h_R=visual_h_R,support_relative=support_relative)
    out=Path(root)/"plots"; out.mkdir(parents=True,exist_ok=True)
    files=data["files"]
    indices=sorted(set((0,len(files)//2,len(files)-1) if snapshot_indices is None else snapshot_indices))
    if any(i<0 or i>=len(files) for i in indices):
        raise ValueError("snapshot index outside saved frame range")
    report={"frames":len(files),"snapshots":[],"movie_status":"not_requested",
            "visual_h_R":data["visual_h_R"],"support_relative":support_relative,
            "field_color_limits":{k:data["limits"][k] for k in ("a","b","delta_b")},
            "joint_density_estimator":"sum_I eta_I Gaussian_h(R-R_I) |Lambda_I(q)|^2; visualization only",
            "unavailable_by_design":["exact epsilon1/epsilon2", "CT-corrected quantum momentum without backend prescription"]}
    if data["config"].get("method") == "ctmqc":
        report["unavailable_by_design"] = ["exact epsilon1/epsilon2 without complex chi/gauge closure"]
        report["ct_quantum_momentum"] = "See channels/ and Q_raw/Q_pairs frame fields; pair values are not one scalar field"
    with plt.rc_context(STYLE):
        fig=plot_summary(data); fig.savefig(out/"summary.png",dpi=dpi); plt.close(fig)
        for i in indices:
            fig=plot_snapshot(_load_frame(files[i]),data)
            target=out/("snapshot_%06d.png"%i)
            fig.savefig(target,dpi=dpi); plt.close(fig)
            report["snapshots"].append(target.name)
        if movies:
            ffmpeg=shutil.which("ffmpeg")
            if ffmpeg is None:
                report["movie_status"]="ffmpeg_unavailable"
                warnings.warn("ffmpeg unavailable; PNG diagnostics are complete")
            else:
                with tempfile.TemporaryDirectory(prefix="gaussian_movie_",dir=str(out)) as temporary:
                    for i,path in enumerate(files):
                        fig=plot_snapshot(_load_frame(path),data)
                        fig.savefig(Path(temporary)/("frame_%06d.png"%i),dpi=movie_dpi); plt.close(fig)
                    command=[ffmpeg,"-y","-loglevel","error","-framerate",str(fps),"-i",
                             str(Path(temporary)/"frame_%06d.png"),"-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2",
                             "-c:v","libx264","-crf","23","-pix_fmt","yuv420p",str(out/"diagnostics.mp4")]
                    result=subprocess.run(command,capture_output=True,text=True)
                    report["movie_status"]="complete" if result.returncode==0 else "encoder_failed"
                    if result.returncode:
                        report["movie_error"]=result.stderr[-3000:]
                        warnings.warn("Movie encoder failed; PNG diagnostics are preserved")
    (out/"render_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--movies",action="store_true")
    p.add_argument("--visual-h-R",type=float)
    p.add_argument("--support-relative",type=float,default=1e-5)
    p.add_argument("--dpi",type=int,default=130)
    a=p.parse_args(argv)
    print(json.dumps(render(a.root,movies=a.movies,visual_h_R=a.visual_h_R,
                            support_relative=a.support_relative,dpi=a.dpi),indent=2))


if __name__=="__main__":
    main()
