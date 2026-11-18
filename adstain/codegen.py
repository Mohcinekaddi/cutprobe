import os
import re

from .onnx_graph import data_inputs, extract_submodel, tensor_info


class Stage:
    def __init__(self, rank, resource, nodes, onnx_path):
        self.rank = rank
        self.resource = resource
        self.nodes = nodes
        self.onnx_path = onnx_path
        self.inbuffs = []
        self.recv = {}
        self.send = {}
        self.outbuffs = []
        self.related = {}


def mapping_from_cuts(model, cuts, placement):
    names = [n.name for n in model.graph.node]
    n = len(names)
    cuts = [int(c) for c in cuts]
    if len(cuts) != len(placement):
        raise ValueError(f"{len(cuts)} cut value(s) for {len(placement)} resource(s): give one "
                         f"value per resource, the first being 0")
    if not cuts or cuts[0] != 0:
        raise ValueError(f"the first cut must be 0 (got {cuts})")
    if sorted(set(cuts)) != cuts:
        raise ValueError(f"cuts must be strictly increasing (got {cuts})")
    if cuts[-1] >= n:
        raise ValueError(f"cut {cuts[-1]} is past the last layer: this model has {n} layers "
                         f"(0..{n - 1}), see 'inspect --layers'")
    if len(set(placement)) != len(placement):
        raise ValueError(f"a resource appears twice in {placement}")
    bounds = cuts + [len(names)]
    mapping = {}
    for s, res in enumerate(placement):
        seg = names[bounds[s]:bounds[s + 1]]
        if not seg:
            raise ValueError(f"empty segment {s} for cuts {cuts}")
        mapping[res] = seg
    return mapping


def _upstream_inputs(graph, tensor, inbuffs):
    producer = {o: n for n in graph.node for o in n.output}
    targets = set(inbuffs)
    found, seen, stack = set(), set(), [tensor]
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        if t in targets:
            found.add(t)
            continue
        node = producer.get(t)
        if node is not None:
            stack.extend(x for x in node.input if x)
    return [b for b in inbuffs if b in found]


def build_stages(model, mapping, resources, outdir):
    g = model.graph
    res = {r.name: r for r in resources}
    node_rank = {}
    for r, nodes in enumerate(mapping.values()):
        for n in nodes:
            node_rank[n] = r
    names = [n.name for n in g.node]
    missing = set(names) - set(node_rank)
    unknown = set(node_rank) - set(names)
    if missing or unknown:
        raise ValueError(f"mapping inconsistent with model: {len(missing)} unmapped node(s), "
                         f"{len(unknown)} unknown node(s)")

    tensor_rank, order = {}, {}
    for k, n in enumerate(g.node):
        for o in n.output:
            tensor_rank[o] = node_rank[n.name]
            order[o] = k
    graph_in = set(data_inputs(g))
    finals = [o.name for o in g.output]

    stages, subs = [], []
    for r, (name, nodes) in enumerate(mapping.items()):
        if name not in res:
            raise ValueError(f"unknown resource {name!r}")
        path = os.path.join(outdir, res[name].ident + ".onnx")
        sub = extract_submodel(model, nodes, path)
        st = Stage(r, res[name], nodes, path)
        st.inbuffs = data_inputs(sub.graph)
        stages.append(st)
        subs.append(sub.graph)

    for st in stages:
        for b in st.inbuffs:
            if b in graph_in:
                continue
            src = tensor_rank[b]
            st.recv[b] = src
            stages[src].send.setdefault(b, []).append(st.rank)
    for st, sub in zip(stages, subs):
        outs = set(st.send) | {o for o in finals if tensor_rank.get(o) == st.rank}
        st.outbuffs = sorted(outs, key=lambda t: order[t])
        st.related = {o: _upstream_inputs(sub, o, st.inbuffs) for o in st.outbuffs}
    return stages


def write_rankfile(path, stages):
    with open(path, "w") as fh:
        for st in stages:
            fh.write(f"rank {st.rank}={st.resource.host} slots={st.resource.slots}\n")


def write_hostfile(path, stages):
    with open(path, "w") as fh:
        for host in dict.fromkeys(st.resource.host for st in stages):
            fh.write(host + "\n")


def _c_str(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


class _Vars:
    def __init__(self):
        self.names = {}
        self.used = set()

    def __getitem__(self, tensor):
        if tensor not in self.names:
            base = "b_" + re.sub(r"\W", "_", tensor)
            name, k = base, 1
            while name in self.used:
                name, k = f"{base}_{k}", k + 1
            self.used.add(name)
            self.names[tensor] = name
        return self.names[tensor]


def _ncnn_dims(dims):
    dims = list(dims[1:])
    if not dims or any(d <= 0 for d in dims):
        return None
    if len(dims) > 3:
        raise ValueError(f"tensor of rank {len(dims) + 1} cannot cross a partition (ncnn Mat)")
    return dims[::-1]


_HEADER = r"""#include "net.h"
#include "benchmark.h"
#include "cpu.h"
#include "gpu.h"
#include <mpi.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static bool env_flag(const char* name, bool dflt)
{
    const char* v = getenv(name);
    if (!v || !v[0]) return dflt;
    return !(v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N');
}

static const bool OPT_FP16 = env_flag("NCNN_FP16", %(fp16)s);
static const bool OPT_FP16_ARITH = env_flag("NCNN_FP16_ARITHMETIC", %(fp16)s);
static const bool OPT_WINOGRAD = env_flag("NCNN_WINOGRAD", %(winograd)s);
static const bool OPT_SGEMM = env_flag("NCNN_SGEMM", true);
static const bool OPT_INT8 = env_flag("NCNN_INT8", %(int8)s);
static const bool OPT_PACKING = env_flag("NCNN_PACKING", true);

static double rss_mb()
{
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return -1.0;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof(line), f))
    {
        if (strncmp(line, "VmRSS:", 6) == 0)
        {
            kb = atol(line + 6);
            break;
        }
    }
    fclose(f);
    return kb / 1024.0;
}

static double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n %% 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void write_metric(int rank, const char* what, double value)
{
    char path[64];
    snprintf(path, sizeof(path), "%%d_%%s.txt", rank, what);
    FILE* f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "%%.4f\n", value);
    fclose(f);
}

static void save_mat(const ncnn::Mat& m, const char* path)
{
    FILE* f = fopen(path, "wb");
    if (!f) return;
    int hdr[4] = {m.dims, m.w, m.h, m.c};
    fwrite(hdr, sizeof(int), 4, f);
    for (int q = 0; q < m.c; q++)
        fwrite((const float*)m.channel(q), sizeof(float), (size_t)m.w * m.h, f);
    fclose(f);
}

static void fail(const char* what, const char* name)
{
    fprintf(stderr, "[bench] %%s: %%s\n", what, name);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

static void set_options(ncnn::Net& net, bool gpu, int threads, int device)
{
    net.opt.blob_allocator = &g_blob_pool_allocator;
    net.opt.workspace_allocator = &g_workspace_pool_allocator;
    net.opt.use_vulkan_compute = gpu;
    if (gpu) net.set_vulkan_device(device);
    net.opt.use_winograd_convolution = OPT_WINOGRAD;
    net.opt.use_sgemm_convolution = OPT_SGEMM;
    net.opt.use_int8_inference = OPT_INT8;
    net.opt.use_int8_storage = OPT_INT8;
    net.opt.use_int8_arithmetic = OPT_INT8;
    net.opt.use_fp16_packed = OPT_FP16;
    net.opt.use_fp16_storage = OPT_FP16;
    net.opt.use_fp16_arithmetic = OPT_FP16_ARITH;
    net.opt.use_packing_layout = OPT_PACKING;
    net.opt.use_shader_pack8 = false;
    net.opt.use_image_storage = false;
    net.opt.num_threads = threads;
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(threads);
}
"""


def _c_float(v):
    s = f"{float(v):.9g}"
    if not any(ch in s for ch in ".enN"):
        s += ".0"
    return s + "f"


def _read_input(pp, input_hw):
    h, w = input_hw
    scale = float(pp["pixel_scale"])
    mean = [float(m) / scale for m in pp["mean"]]
    norm = [scale / float(s) for s in pp["std"]]
    pixel = "ncnn::Mat::PIXEL_BGR2RGB" if pp["channel_order"] == "RGB" else "ncnn::Mat::PIXEL_BGR"
    L = ["static int read_input(const char* imagepath, ncnn::Mat& in)", "{",
         "    cv::Mat m = cv::imread(imagepath, cv::IMREAD_COLOR);",
         "    if (m.empty())", "    {",
         '        fprintf(stderr, "cv::imread %s failed\\n", imagepath);',
         "        return -1;", "    }"]
    if pp.get("letterbox"):
        pad = float(pp.get("pad_value", 114))
        L += [f"    const double r = std::min((double){h} / m.rows, (double){w} / m.cols);",
              "    const int nw = (int)std::floor(m.cols * r + 0.5), nh = (int)std::floor(m.rows * r + 0.5);",
              "    cv::Mat resized = m;",
              "    if (nw != m.cols || nh != m.rows)",
              "        cv::resize(m, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);",
              f"    cv::Mat canvas({h}, {w}, CV_8UC3, cv::Scalar({pad:g}, {pad:g}, {pad:g}));",
              f"    resized.copyTo(canvas(cv::Rect(({w} - nw) / 2, ({h} - nh) / 2, nw, nh)));",
              f"    in = ncnn::Mat::from_pixels(canvas.data, {pixel}, {w}, {h});"]
    elif pp.get("resize_short"):
        s = int(pp["resize_short"])
        L += [f"    const int short_side = {s};",
              "    int rw, rh;",
              "    if (m.cols < m.rows)",
              "    {",
              "        rw = short_side;",
              "        rh = (int)((float)m.rows * short_side / m.cols + 0.5f);",
              "    }",
              "    else",
              "    {",
              "        rh = short_side;",
              "        rw = (int)((float)m.cols * short_side / m.rows + 0.5f);",
              "    }",
              "    cv::Mat resized;",
              "    cv::resize(m, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);",
              f"    cv::Mat crop = resized(cv::Rect((rw - {w}) / 2, (rh - {h}) / 2, {w}, {h})).clone();",
              f"    in = ncnn::Mat::from_pixels(crop.data, {pixel}, {w}, {h});"]
    else:
        L += [f"    in = ncnn::Mat::from_pixels_resize(m.data, {pixel}, m.cols, m.rows, {w}, {h});"]
    L += ["    const float mean_vals[3] = {%s};" % ", ".join(_c_float(v) for v in mean),
          "    const float norm_vals[3] = {%s};" % ", ".join(_c_float(v) for v in norm),
          "    in.substract_mean_normalize(mean_vals, norm_vals);",
          "    return 0;", "}", ""]
    return L


_HOOKS = r"""
static int g_img_offset = 0;

static bool env_on(const char* name)
{
    const char* v = getenv(name);
    return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T');
}

static FILE* fm_open(const char* filename, const char* mode)
{
    const char* dir = getenv("FM_DIR");
    char path[2048];
    if (dir && dir[0])
    {
        mkdir(dir, 0775);
        snprintf(path, sizeof(path), "%s/%s", dir, filename);
    }
    else
        snprintf(path, sizeof(path), "%s", filename);
    FILE* f = fopen(path, mode);
    if (!f) fprintf(stderr, "[fm] cannot open %s\n", path);
    else fprintf(stderr, "[fm] %s %s\n", mode[0] == 'w' ? "dumping to" : "attack payload", path);
    return f;
}

static void append_record(FILE* f, const ncnn::Mat& m, int idx)
{
    int hdr[4] = {idx, m.dims >= 3 ? m.c : 1, m.dims >= 2 ? m.h : 1, m.w};
    fwrite(hdr, sizeof(int), 4, f);
    for (int q = 0; q < hdr[1]; q++)
    {
        const ncnn::Mat ch = m.channel(q);
        for (int y = 0; y < hdr[2]; y++)
            fwrite(ch.row(y), sizeof(float), m.w, f);
    }
}

static int load_record(FILE* f, ncnn::Mat& m, int idx)
{
    const int c = m.dims >= 3 ? m.c : 1, h = m.dims >= 2 ? m.h : 1, w = m.w;
    const long rec = 16L + 4L * (long)c * h * w;
    if (fseek(f, (long)(idx - g_img_offset) * rec, SEEK_SET) != 0) return -1;
    int hdr[4];
    if (fread(hdr, sizeof(int), 4, f) != 4) return -1;
    if (hdr[0] != idx || hdr[1] != c || hdr[2] != h || hdr[3] != w) return -1;
    ncnn::Mat out = m.clone();
    for (int q = 0; q < c; q++)
    {
        ncnn::Mat ch = out.channel(q);
        for (int y = 0; y < h; y++)
            if (fread(ch.row(y), sizeof(float), w, f) != (size_t)w) return -1;
    }
    m = out;
    return 0;
}

static FILE* open_results()
{
    const char* p = getenv("RESULTS_FILE");
    FILE* f = fopen(p && p[0] ? p : "results.csv", "w");
    if (f) fprintf(f, "index,top1,top2,top3,score1,score2,score3\n");
    return f;
}
"""


def _report(labels):
    L = ["static void report_scores(const ncnn::Mat& out, int idx, FILE* res)", "{",
         "    static std::vector<std::string> labels;",
         "    static bool loaded = false;"]
    if labels:
        L += ["    if (!loaded)", "    {",
              f"        FILE* lf = fopen({_c_str(labels)}, \"r\");",
              "        if (lf)", "        {", "            char buf[1024];",
              "            while (fgets(buf, sizeof(buf), lf))", "            {",
              "                std::string s(buf);",
              "                while (!s.empty() && (s.back() == '\\n' || s.back() == '\\r')) s.pop_back();",
              "                size_t sp = s.find(' ');",
              "                labels.push_back(sp == std::string::npos ? s : s.substr(sp + 1));",
              "            }", "            fclose(lf);", "        }", "    }"]
    L += ["    loaded = true;",
          "    ncnn::Mat flat = out.reshape(out.w * out.h * out.c);",
          "    std::vector<std::pair<float, int> > v(flat.w);",
          "    for (int j = 0; j < flat.w; j++) v[j] = std::make_pair(flat[j], j);",
          "    const int k = std::min(3, flat.w);",
          "    std::partial_sort(v.begin(), v.begin() + k, v.end(), std::greater<std::pair<float, int> >());",
          "    const int top = k ? v[0].second : -1;",
          "    const char* name = top >= 0 && top < (int)labels.size() ? labels[top].c_str() : \"\";",
          "    if (!res || idx < 3 || idx % 1000 == 0)",
          '        fprintf(stderr, "[%d] top1 = %d (%f) %s\\n", idx, top, k ? v[0].first : 0.f, name);',
          "    if (!res) return;",
          '    fprintf(res, "%d", idx);',
          "    for (int i = 0; i < 3; i++) fprintf(res, \",%d\", i < k ? v[i].second : -1);",
          "    for (int i = 0; i < 3; i++) fprintf(res, \",%g\", i < k ? v[i].first : 0.f);",
          '    fprintf(res, "\\n");', "}", ""]
    return L


_DET_COMMON = r"""
struct Det
{
    float x1, y1, x2, y2, score;
    int label;
};

static float env_float(const char* name, float dflt)
{
    const char* v = getenv(name);
    return v && v[0] ? (float)atof(v) : dflt;
}

static const float DET_CONF = env_float("DET_CONF", %(conf)s);
static const float DET_IOU = env_float("DET_IOU", %(iou)s);
static const int DET_MAX = (int)env_float("DET_MAX", %(max_det)d);
static const int DET_MAX_NMS = %(max_nms)d;
static const bool DET_AGNOSTIC = env_flag("DET_AGNOSTIC", %(agnostic)s);

static inline float sigmoid_f(float x)
{
    return 1.f / (1.f + std::exp(-x));
}

static float det_iou(const Det& a, const Det& b)
{
    const float iw = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    const float ih = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
    if (iw <= 0.f || ih <= 0.f) return 0.f;
    const float inter = iw * ih;
    const float ua = std::max(a.x2 - a.x1, 0.f) * std::max(a.y2 - a.y1, 0.f)
                     + std::max(b.x2 - b.x1, 0.f) * std::max(b.y2 - b.y1, 0.f) - inter;
    return ua > 0.f ? inter / ua : 0.f;
}

static bool det_cmp(const Det& a, const Det& b)
{
    return a.score > b.score;
}

static void det_select(std::vector<Det>& cand, std::vector<Det>& out, bool nms)
{
    std::stable_sort(cand.begin(), cand.end(), det_cmp);
    if ((int)cand.size() > DET_MAX_NMS) cand.resize(DET_MAX_NMS);
    out.clear();
    if (!nms)
    {
        for (size_t i = 0; i < cand.size() && (int)out.size() < DET_MAX; i++) out.push_back(cand[i]);
        return;
    }
    std::vector<char> gone(cand.size(), 0);
    for (size_t i = 0; i < cand.size() && (int)out.size() < DET_MAX; i++)
    {
        if (gone[i]) continue;
        out.push_back(cand[i]);
        for (size_t j = i + 1; j < cand.size(); j++)
            if (!gone[j] && (DET_AGNOSTIC || cand[j].label == cand[i].label) && det_iou(cand[i], cand[j]) > DET_IOU)
                gone[j] = 1;
    }
}

static FILE* open_detections()
{
    const char* p = getenv("RESULTS_FILE");
    FILE* f = fopen(p && p[0] ? p : "results.csv", "w");
    if (f) fprintf(f, "index,x1,y1,x2,y2,score,class\n");
    return f;
}
"""


def _det_decoder(head, input_hw):
    fmt = head["format"]
    nc = int(head.get("num_classes") or 0)
    H, W = input_hw
    L = []
    if fmt == "yolov8":
        R = int(head["reg_max"])
        L += ["static void yolo_box(const float* p, size_t cs, size_t o, float ax, float ay, float stride, Det& t)",
              "{", f"    const int R = {R};",
              "    float d[4];",
              "    for (int s = 0; s < 4; s++)", "    {",
              "        if (R == 1)", "        {", "            d[s] = p[(size_t)s * cs + o];",
              "            continue;", "        }",
              "        float mx = -FLT_MAX;",
              "        for (int j = 0; j < R; j++) mx = std::max(mx, p[(size_t)(s * R + j) * cs + o]);",
              "        float den = 0.f, num = 0.f;",
              "        for (int j = 0; j < R; j++)", "        {",
              "            const float e = std::exp(p[(size_t)(s * R + j) * cs + o] - mx);",
              "            den += e;", "            num += e * j;", "        }",
              "        d[s] = num / den;", "    }",
              "    t.x1 = (ax - d[0]) * stride;", "    t.y1 = (ay - d[1]) * stride;",
              "    t.x2 = (ax + d[2]) * stride;", "    t.y2 = (ay + d[3]) * stride;", "}", ""]
        if head.get("end2end"):
            L += ["struct E2EAnchor", "{", "    const float* pb;", "    const float* pc;",
                  "    size_t cs, ccs, o;", "    int c0;", "    float ax, ay, stride, maxs;", "};", "",
                  "static bool e2e_cmp(const E2EAnchor& a, const E2EAnchor& b)", "{",
                  "    return a.maxs > b.maxs;", "}", "",
                  "static void e2e_level(const ncnn::Mat& mb, const ncnn::Mat& mc, int c0, float stride, std::vector<E2EAnchor>& anchors)",
                  "{", f"    const int NC = {nc};",
                  "    const float* pc = (const float*)mc.data;",
                  "    for (int y = 0; y < mb.h; y++)",
                  "        for (int x = 0; x < mb.w; x++)", "        {",
                  "            const size_t o = (size_t)y * mb.w + x;",
                  "            float bmax = pc[(size_t)c0 * mc.cstep + o];",
                  "            for (int k = 1; k < NC; k++) bmax = std::max(bmax, pc[(size_t)(c0 + k) * mc.cstep + o]);",
                  "            E2EAnchor a = {(const float*)mb.data, pc, mb.cstep, mc.cstep, o, c0, x + 0.5f, y + 0.5f, stride, sigmoid_f(bmax)};",
                  "            anchors.push_back(a);", "        }", "}", "",
                  "static void e2e_select(std::vector<E2EAnchor>& anchors, std::vector<Det>& out)",
                  "{", f"    const int K = {int(head.get('topk', 300))}, NC = {nc};",
                  "    const size_t k1 = std::min((size_t)K, anchors.size());",
                  "    std::partial_sort(anchors.begin(), anchors.begin() + k1, anchors.end(), e2e_cmp);",
                  "    std::vector<Det> pairs;",
                  "    pairs.reserve(k1 * NC);",
                  "    for (size_t i = 0; i < k1; i++)", "    {",
                  "        const E2EAnchor& a = anchors[i];",
                  "        Det b;",
                  "        yolo_box(a.pb, a.cs, a.o, a.ax, a.ay, a.stride, b);",
                  "        for (int c = 0; c < NC; c++)", "        {",
                  "            Det t = b;",
                  "            t.score = sigmoid_f(a.pc[(size_t)(a.c0 + c) * a.ccs + a.o]);",
                  "            t.label = c;",
                  "            pairs.push_back(t);", "        }", "    }",
                  "    const size_t k2 = std::min((size_t)K, pairs.size());",
                  "    std::partial_sort(pairs.begin(), pairs.begin() + k2, pairs.end(), det_cmp);",
                  "    out.clear();",
                  "    for (size_t i = 0; i < k2 && (int)out.size() < DET_MAX; i++)",
                  "        if (pairs[i].score > DET_CONF) out.push_back(pairs[i]);", "}", ""]
        else:
            L += ["static void decode_level(const ncnn::Mat& mb, const ncnn::Mat& mc, int c0, float stride, std::vector<Det>& cand)",
                  "{", f"    const int NC = {nc};",
                  "    const float* pc = (const float*)mc.data;",
                  "    const size_t ccs = mc.cstep;",
                  "    for (int y = 0; y < mb.h; y++)",
                  "        for (int x = 0; x < mb.w; x++)", "        {",
                  "            const size_t o = (size_t)y * mb.w + x;",
                  "            int best = 0;",
                  "            float bmax = pc[(size_t)c0 * ccs + o];",
                  "            for (int k = 1; k < NC; k++)", "            {",
                  "                const float v = pc[(size_t)(c0 + k) * ccs + o];",
                  "                if (v > bmax)", "                {", "                    bmax = v;",
                  "                    best = k;", "                }", "            }",
                  "            const float score = sigmoid_f(bmax);",
                  "            if (!(score > DET_CONF)) continue;",
                  "            Det t;",
                  "            yolo_box((const float*)mb.data, mb.cstep, o, x + 0.5f, y + 0.5f, stride, t);",
                  "            t.score = score;", "            t.label = best;",
                  "            cand.push_back(t);", "        }", "}", ""]
    elif fmt == "yolov5":
        na = len(head["anchors"][0])
        L += ["static void decode_level(const ncnn::Mat& m, float stride, const float* anchors, std::vector<Det>& cand)",
              "{", f"    const int NA = {na}, NO = {nc + 5}, NC = {nc};",
              "    const float* p = (const float*)m.data;",
              "    const size_t cs = m.cstep;",
              "    for (int a = 0; a < NA; a++)",
              "        for (int y = 0; y < m.h; y++)",
              "            for (int x = 0; x < m.w; x++)", "            {",
              "                const size_t o = (size_t)y * m.w + x;",
              "                const float* q = p + (size_t)(a * NO) * cs + o;",
              "                const float obj = sigmoid_f(q[4 * cs]);",
              "                if (!(obj > DET_CONF)) continue;",
              "                int best = 0;",
              "                float bmax = q[5 * cs];",
              "                for (int k = 1; k < NC; k++)", "                {",
              "                    const float v = q[(size_t)(5 + k) * cs];",
              "                    if (v > bmax)", "                    {", "                        bmax = v;",
              "                        best = k;", "                    }", "                }",
              "                const float score = obj * sigmoid_f(bmax);",
              "                if (!(score > DET_CONF)) continue;",
              "                const float cx = (sigmoid_f(q[0]) * 2.f - 0.5f + x) * stride;",
              "                const float cy = (sigmoid_f(q[cs]) * 2.f - 0.5f + y) * stride;",
              "                const float sw = sigmoid_f(q[2 * cs]) * 2.f, sh = sigmoid_f(q[3 * cs]) * 2.f;",
              "                const float bw = sw * sw * anchors[2 * a], bh = sh * sh * anchors[2 * a + 1];",
              "                Det t;",
              "                t.x1 = cx - bw * 0.5f;", "                t.y1 = cy - bh * 0.5f;",
              "                t.x2 = cx + bw * 0.5f;", "                t.y2 = cy + bh * 0.5f;",
              "                t.score = score;", "                t.label = best;",
              "                cand.push_back(t);", "            }", "}", ""]
    else:
        box_fmt = head.get("box_format") or ("cxcywh" if fmt == "yolo_decoded" else "xyxy")
        sx, sy = (_c_float(W), _c_float(H)) if head.get("normalized") else (_c_float(1), _c_float(1))
        L += ["static void push_box(std::vector<Det>& cand, float a, float b, float c, float d, float score, int label)",
              "{", "    Det t;"]
        if box_fmt == "cxcywh":
            L += [f"    t.x1 = (a - c * 0.5f) * {sx};", f"    t.y1 = (b - d * 0.5f) * {sy};",
                  f"    t.x2 = (a + c * 0.5f) * {sx};", f"    t.y2 = (b + d * 0.5f) * {sy};"]
        elif box_fmt == "xywh":
            L += [f"    t.x1 = a * {sx};", f"    t.y1 = b * {sy};",
                  f"    t.x2 = (a + c) * {sx};", f"    t.y2 = (b + d) * {sy};"]
        else:
            L += [f"    t.x1 = a * {sx};", f"    t.y1 = b * {sy};",
                  f"    t.x2 = c * {sx};", f"    t.y2 = d * {sy};"]
        L += ["    t.score = score;", "    t.label = label;", "    cand.push_back(t);", "}", ""]
    return L


def _det_decode_calls(head, var_of):
    fmt = head["format"]
    outs = head["outputs"]
    nc = int(head.get("num_classes") or 0)
    L = []
    if fmt == "yolov8":
        R = int(head["reg_max"])
        pairs = list(zip(outs[0::2], outs[1::2])) if head.get("split") else [(t, t) for t in outs]
        c0 = 0 if head.get("split") else 4 * R
        if head.get("end2end"):
            L.append("    std::vector<E2EAnchor> anchors;")
            for (tb, tc), s in zip(pairs, head["strides"]):
                L.append(f"    e2e_level({var_of(tb)}, {var_of(tc)}, {c0}, {_c_float(s)}, anchors);")
            L.append("    e2e_select(anchors, dets);")
            return L
        for (tb, tc), s in zip(pairs, head["strides"]):
            L.append(f"    decode_level({var_of(tb)}, {var_of(tc)}, {c0}, {_c_float(s)}, cand);")
        L.append("    const bool nms = true;")
    elif fmt == "yolov5":
        for k, (t, s, anc) in enumerate(zip(outs, head["strides"], head["anchors"])):
            vals = ", ".join(_c_float(v) for a in anc for v in a)
            L.append(f"    static const float anchors{k}[] = {{{vals}}};")
            L.append(f"    decode_level({var_of(t)}, {_c_float(s)}, anchors{k}, cand);")
        L.append("    const bool nms = true;")
    elif fmt == "yolo_decoded":
        obj = bool(head.get("objectness"))
        first = head.get("layout") == "anchors_first"
        k0 = 5 if obj else 4
        L += [f"    const ncnn::Mat f = {var_of(outs[0])}.reshape({var_of(outs[0])}.w * {var_of(outs[0])}.h * {var_of(outs[0])}.c);",
              "    const float* p = (const float*)f.data;",
              f"    const int K = {nc + k0}, A = f.w / K;"]
        at = (lambda i, k: f"p[(size_t)({i}) * K + ({k})]") if first else (lambda i, k: f"p[(size_t)({k}) * A + ({i})]")
        L += ["    for (int i = 0; i < A; i++)", "    {",
              f"        int best = 0;", f"        float bmax = {at('i', k0)};",
              f"        for (int k = 1; k < {nc}; k++)", "        {",
              f"            const float v = {at('i', f'{k0} + k')};",
              "            if (v > bmax)", "            {", "                bmax = v;", "                best = k;",
              "            }", "        }",
              f"        const float score = bmax{' * ' + at('i', 4) if obj else ''};",
              "        if (!(score > DET_CONF)) continue;",
              f"        push_box(cand, {at('i', 0)}, {at('i', 1)}, {at('i', 2)}, {at('i', 3)}, score, best);",
              "    }", "    const bool nms = true;"]
    elif fmt == "boxes_scores":
        first = head.get("layout") == "anchors_first"
        bg = head.get("background")
        L += [f"    const ncnn::Mat fb = {var_of(outs[0])}.reshape({var_of(outs[0])}.w * {var_of(outs[0])}.h * {var_of(outs[0])}.c);",
              f"    const ncnn::Mat fs = {var_of(outs[1])}.reshape({var_of(outs[1])}.w * {var_of(outs[1])}.h * {var_of(outs[1])}.c);",
              "    const float* b = (const float*)fb.data;",
              "    const float* s = (const float*)fs.data;",
              f"    const int A = fb.w / 4, C = {nc};"]
        sat = (lambda i, c: f"s[(size_t)({i}) * C + ({c})]") if first else (lambda i, c: f"s[(size_t)({c}) * A + ({i})]")
        L += ["    for (int i = 0; i < A; i++)", "    {", "        int best = -1;", "        float bmax = -FLT_MAX;",
              "        for (int c = 0; c < C; c++)", "        {"]
        if bg is not None:
            L.append(f"            if (c == {int(bg)}) continue;")
        L += [f"            const float v = {sat('i', 'c')};",
              "            if (v > bmax)", "            {", "                bmax = v;", "                best = c;",
              "            }", "        }",
              "        if (best < 0 || !(bmax > DET_CONF)) continue;",
              "        push_box(cand, b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3], bmax, best);",
              "    }", "    const bool nms = true;"]
    elif fmt == "end2end":
        if head.get("layout") == "split":
            L += [f"    const ncnn::Mat fb = {var_of(outs[0])}.reshape({var_of(outs[0])}.w * {var_of(outs[0])}.h * {var_of(outs[0])}.c);",
                  f"    const ncnn::Mat fs = {var_of(outs[1])}.reshape({var_of(outs[1])}.w * {var_of(outs[1])}.h * {var_of(outs[1])}.c);",
                  "    const float* b = (const float*)fb.data;",
                  "    const float* s = (const float*)fs.data;"]
            if outs[2]:
                L += [f"    const ncnn::Mat fl = {var_of(outs[2])}.reshape({var_of(outs[2])}.w * {var_of(outs[2])}.h * {var_of(outs[2])}.c);",
                      "    const float* l = (const float*)fl.data;"]
            L += ["    for (int i = 0; i < fs.w; i++)",
                  "        if (s[i] > DET_CONF)",
                  "            push_box(cand, b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3], s[i], "
                  + ("(int)l[i]" if outs[2] else "0") + ");"]
        else:
            L += [f"    const ncnn::Mat f = {var_of(outs[0])}.reshape({var_of(outs[0])}.w * {var_of(outs[0])}.h * {var_of(outs[0])}.c);",
                  "    const float* p = (const float*)f.data;",
                  f"    const int K = {var_of(outs[0])}.w, N = f.w / K;",
                  "    for (int i = 0; i < N; i++)",
                  "        if (p[i * K + 4] > DET_CONF)",
                  "            push_box(cand, p[i * K], p[i * K + 1], p[i * K + 2], p[i * K + 3], p[i * K + 4], (int)p[i * K + 5]);"]
        L.append("    const bool nms = false;")
    else:
        raise ValueError(f"no C++ decoder for detection head {fmt!r}")
    return L + ["    det_select(cand, dets, nms);"]


def _report_det(head, det, names, input_hw, finals, var):
    flags = {"conf": _c_float(det["eval_conf"]), "iou": _c_float(det["iou"]),
             "max_det": int(det["max_det"]), "max_nms": int(det["max_nms"]),
             "agnostic": "true" if det["agnostic_nms"] else "false"}
    L = (_DET_COMMON % flags).split("\n")
    L += _det_decoder(head, input_hw)
    if names:
        L.append("static const char* const CLASS_NAMES[] = {" + ", ".join(_c_str(n) for n in names) + "};")
        L.append(f"static const int N_NAMES = {len(names)};")
    else:
        L += ["static const char* const* CLASS_NAMES = NULL;", "static const int N_NAMES = 0;"]
    args = ", ".join(f"const ncnn::Mat& {var[f]}" for f in finals)
    L += ["", f"static void report_detections({args}, int idx, FILE* res)", "{",
          "    std::vector<Det> cand, dets;"]
    L += _det_decode_calls(head, lambda t: var[t])
    L += ["    if (!res || idx < 3 || idx % 1000 == 0)", "    {",
          "        if (dets.empty())",
          '            fprintf(stderr, "[%d] no detection\\n", idx);',
          "        else", "        {",
          '            fprintf(stderr, "[%d] %d detection(s) above %g:", idx, (int)dets.size(), DET_CONF);',
          "            for (size_t i = 0; i < dets.size() && i < 5; i++)", "            {",
          "                const int c = dets[i].label;",
          '                fprintf(stderr, " %s %.3f [%.0f %.0f %.0f %.0f]", c >= 0 && c < N_NAMES ? CLASS_NAMES[c] : "?", '
          "dets[i].score, dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2);",
          "            }", '            fprintf(stderr, "\\n");', "        }", "    }",
          "    if (!res) return;",
          "    if (dets.empty())",
          '        fprintf(res, "%d,0,0,0,0,0,-1\\n", idx);',
          "    for (size_t i = 0; i < dets.size(); i++)",
          '        fprintf(res, "%d,%.2f,%.2f,%.2f,%.2f,%.5f,%d\\n", idx, dets[i].x1, dets[i].y1, dets[i].x2, '
          "dets[i].y2, dets[i].score, dets[i].label);",
          "}", ""]
    return L


def buffer_files(stages):
    counts = {}
    for st in stages:
        for b in st.recv:
            counts[b] = counts.get(b, 0) + 1
    out, used = {}, set()
    for st in stages:
        for b in st.recv:
            base = re.sub(r"[^0-9a-zA-Z_]", "_", b) or "buf"
            if counts[b] > 1:
                base += f"_r{st.rank}"
            name, k = base, 2
            while name in used:
                name, k = f"{base}_{k}", k + 1
            used.add(name)
            out[(st.rank, b)] = name
    return out


def buffer_manifest(stages, model, cuts):
    info = tensor_info(model.graph)
    files = buffer_files(stages)
    out = []
    for st in stages:
        for b in st.recv:
            safe = files[(st.rank, b)]
            out.append({"rank": st.rank, "resource": st.resource.name, "buffer_name": b,
                        "safe_name": safe, "source_rank": st.recv[b],
                        "cut": int(cuts[st.rank]),
                        "shape": info[b][1] if b in info else None,
                        "filename": f"feature_maps_buffer_{safe}.bin",
                        "attack_filename": f"attacked_feature_maps_{safe}.bin",
                        "attack_target_layer": st.rank + 1})
    return out


def _rank(st, graph_input, input_hw, info, var, tags, files, opts):
    r = st.rank
    res = st.resource
    infer = opts["mode"] == "infer"
    marker = opts["marker"]
    dummy = lambda b: bool(marker) and marker.upper() in b.upper()
    recvs = [b for b in st.inbuffs if b in st.recv and not dummy(b)]
    sends = [(b, d) for b in st.outbuffs for d in st.send.get(b, []) if not dummy(b)]
    hooked = recvs if infer else []
    req = {}
    for b, d in sends:
        req[("s", b, d)] = len(req)
    for b in recvs:
        req[("r", b)] = len(req)
    n_req = max(len(req), 1)
    final = all(f in st.outbuffs for f in opts["finals"])
    if opts["detect"]:
        open_call = "open_detections()"
        mats = ", ".join(var[f] for f in opts["finals"])
        report = lambda i, res: f"report_detections({mats}, {i}, {res});"
    else:
        open_call = "open_results()"
        report = lambda i, res: f"report_scores({var[opts['final']]}, {i}, {res});"

    def numel(b):
        dims = _ncnn_dims(info[b][1])
        if dims is None:
            raise ValueError(f"unknown static shape for crossing tensor {b!r}")
        n = 1
        for d in dims:
            n *= d
        return n, dims

    bufs = st.inbuffs + st.outbuffs
    L = [f"static ncnn::Net g_net{r};"]
    for b in hooked:
        L += [f"static FILE* g_dump_{files[(r, b)]} = NULL;",
              f"static FILE* g_atk_{files[(r, b)]} = NULL;"]
    L += ["", f"static void model{r}_init()", "{",
          f"    set_options(g_net{r}, {'true' if res.kind == 'gpu' else 'false'}, {res.threads}, {res.device});",
          f"    if (g_net{r}.load_param({_c_str(res.ident + '.param')}) != 0) fail(\"load_param\", {_c_str(res.ident)});",
          f"    if (g_net{r}.load_model({_c_str(res.ident + '.bin')}) != 0) fail(\"load_model\", {_c_str(res.ident)});"]
    if hooked:
        L += ['    const char* layer = getenv("ATTACK_TARGET_LAYER");',
              f'    const bool attack = env_on("ATTACK_ENABLED") && layer && atoi(layer) == {r + 1};',
              '    const bool dump = env_on("DUMP_BUFFER");']
        for b in hooked:
            f = files[(r, b)]
            L += [f'    if (dump) g_dump_{f} = fm_open("feature_maps_buffer_{f}.bin", "wb");',
                  f'    if (attack) g_atk_{f} = fm_open("attacked_feature_maps_{f}.bin", "rb");']
    L += ["}", "", f"static void model{r}_close()", "{", f"    g_net{r}.clear();"]
    for b in hooked:
        f = files[(r, b)]
        L += [f"    if (g_dump_{f}) fclose(g_dump_{f});", f"    if (g_atk_{f}) fclose(g_atk_{f});"]
    L += ["}", ""]

    args = ["const char* imagepath", "int img_idx"] + [f"ncnn::Mat& {var[b]}" for b in bufs] + ["double& comp"]
    L += [f"static void model{r}_run({', '.join(args)})", "{",
          "    (void)img_idx;",
          f"    MPI_Request req[{n_req}];",
          f"    MPI_Status sts[{n_req}];"]
    if graph_input in st.inbuffs:
        h, w = input_hw
        g = var[graph_input]
        L += [f"    if (imagepath && read_input(imagepath, {g}) != 0)", "    {",
              f"        {g} = ncnn::Mat({w}, {h}, 3);", f"        {g}.fill(0.f);", "    }"]
    for b in recvs:
        L.append(f"    ncnn::Mat r_{var[b]}({numel(b)[0]});")
    for b in dict.fromkeys(b for b, _ in sends):
        L.append(f"    ncnn::Mat s_{var[b]};")
    for b in recvs:
        i = req[("r", b)]
        L.append(f"    MPI_Irecv((float*)r_{var[b]}, {numel(b)[0]}, MPI_FLOAT, {st.recv[b]}, "
                 f"{tags[(st.recv[b], b, r)]}, MPI_COMM_WORLD, &req[{i}]);")
    L.append(f"    ncnn::Extractor ex = g_net{r}.create_extractor();")
    waited = set()

    def wait(b):
        i = req[("r", b)]
        _, dims = numel(b)
        out = [f"    MPI_Wait(&req[{i}], &sts[{i}]);",
               f"    {var[b]} = r_{var[b]};" if len(dims) == 1 else
               f"    {var[b]} = r_{var[b]}.reshape({', '.join(map(str, dims))});"]
        if b in hooked:
            f = files[(r, b)]
            out += [f"    if (g_dump_{f}) append_record(g_dump_{f}, {var[b]}, img_idx);",
                    f"    if (g_atk_{f} && load_record(g_atk_{f}, {var[b]}, img_idx) != 0)",
                    f'        fprintf(stderr, "[attack] img %d: no payload record for {b}, clean buffer kept\\n", img_idx);']
        waited.add(b)
        return out

    for ob in st.outbuffs:
        rel = st.related[ob]
        for b in rel:
            if b in recvs and b not in waited:
                L += wait(b)
        L += ["    {", "        double c0 = ncnn::get_current_time();"]
        for b in rel:
            L.append(f"        ex.input({_c_str(b)}, {var[b]});")
        if not dummy(ob):
            L.append(f"        ex.extract({_c_str(ob)}, {var[ob]});")
        L += ["        comp += ncnn::get_current_time() - c0;", "    }"]
        dests = st.send.get(ob, []) if not dummy(ob) else []
        if dests:
            n, _ = numel(ob)
            L += [f"    s_{var[ob]} = {var[ob]}.reshape({n});",
                  f"    if (s_{var[ob]}.empty()) fail(\"size mismatch on\", {_c_str(ob)});"]
            for d in dests:
                i = req[("s", ob, d)]
                L.append(f"    MPI_Isend((float*)s_{var[ob]}, {n}, MPI_FLOAT, {d}, "
                         f"{tags[(r, ob, d)]}, MPI_COMM_WORLD, &req[{i}]);")
    for b in recvs:
        if b not in waited:
            L += wait(b)
    for b, d in sends:
        i = req[("s", b, d)]
        L.append(f"    MPI_Wait(&req[{i}], &sts[{i}]);")
    L += ["}", ""]

    call = ", ".join(var[b] for b in bufs)
    L += [f"static void rank{r}_main(const char* arg)", "{",
          "    int rank;", "    MPI_Comm_rank(MPI_COMM_WORLD, &rank);"]
    for b in dict.fromkeys(bufs):
        L.append(f"    ncnn::Mat {var[b]};")
    if infer:
        L += ["    std::ifstream list(arg);",
              "    if (!list.is_open()) fail(\"cannot open image list\", arg);"]
        if final:
            L.append(f"    FILE* res = {open_call};")
        L += ["    std::string line;", "    int idx = g_img_offset;", "    double comp = 0.0;",
              "    double t0 = ncnn::get_current_time();",
              "    while (std::getline(list, line))", "    {",
              "        while (!line.empty() && (line.back() == '\\r' || line.back() == ' ')) line.pop_back();",
              "        if (line.empty()) continue;",
              f"        model{r}_run(line.c_str(), idx, {call}, comp);"]
        if final:
            L.append("        " + report("idx", "res"))
        L += ["        idx++;", "    }"]
        if final:
            L.append("    if (res) fclose(res);")
        L += ["    const int n_img = idx - g_img_offset;",
              '    fprintf(stderr, "rank %d: %d image(s), %.2f ms/image (compute %.2f ms)\\n", rank, n_img, '
              "(ncnn::get_current_time() - t0) / (n_img ? n_img : 1), comp / (n_img ? n_img : 1));"]
    else:
        L += [f"    const int warmup = {opts['warmup']};", f"    const int loops = {opts['loops']};",
              "    std::vector<double> walls, comps;",
              "    for (int it = 0; it < warmup + loops; it++)", "    {",
              "        double c = 0.0;",
              "        double t0 = ncnn::get_current_time();",
              f"        model{r}_run(it == 0 ? arg : NULL, 0, {call}, c);",
              "        double dt = ncnn::get_current_time() - t0;",
              "        if (it >= warmup)", "        {", "            walls.push_back(dt);", "            comps.push_back(c);",
              "        }", "    }",
              "    const double wall = median(walls), comp = median(comps);",
              '    write_metric(rank, "PERFORMANCE", wall);',
              '    write_metric(rank, "COMPUTE", comp);',
              '    write_metric(rank, "MEMORY", rss_mb());',
              '    fprintf(stderr, "rank %d: median wall %.2f ms (min %.2f, max %.2f), median compute %.2f ms, rss %.1f MB\\n", '
              "rank, wall, *std::min_element(walls.begin(), walls.end()), *std::max_element(walls.begin(), walls.end()), comp, rss_mb());"]
        if final:
            L.append("    " + report("0", "NULL"))
        if opts["dump"]:
            for b in recvs:
                L.append(f"    save_mat({var[b]}, {_c_str('fm_' + re.sub(r'[^A-Za-z0-9_.-]', '_', b) + '.bin')});")
    L += ["}", ""]
    return L


def generate_cpp(stages, model, preprocess, input_hw, opts):
    g = model.graph
    info = tensor_info(g)
    graph_input = data_inputs(g)[0]
    task = opts.get("task")
    detect = bool(task is not None and task.detection)
    finals = [o.name for o in g.output] if detect else [g.output[0].name]
    holders = [st.rank for st in stages if all(f in st.outbuffs for f in finals)]
    if len(holders) != 1:
        raise ValueError(f"the graph outputs {finals} are not all computed by one rank: move the last "
                         f"cut before them (see 'inspect --layers')")
    opts = dict(opts, final=finals[0], finals=finals, detect=detect)
    var = _Vars()
    files = buffer_files(stages)
    tags = {}
    for st in stages:
        for b, dests in st.send.items():
            for d in dests:
                tags[(st.rank, b, d)] = len(tags)

    det = opts["deterministic"]
    flags = {"fp16": "false" if det else "true", "winograd": "false" if det else "true",
             "int8": "false"}
    L = (_HEADER % flags).split("\n")
    L.insert(L.index("#include <vector>") + 1, "#include <fstream>")
    L.insert(L.index("#include <fstream>") + 1, "#include <sys/stat.h>")
    L += _HOOKS.split("\n")
    L += _read_input(preprocess, input_hw)
    if detect:
        L.insert(L.index("#include <vector>") + 1, "#include <cmath>")
        L += _report_det(task.head, opts["det"], task.names, input_hw, finals, var)
    else:
        L += _report(opts.get("labels"))
    for st in stages:
        L += _rank(st, graph_input, input_hw, info, var, tags, files, opts)

    gpu_ranks = [st.rank for st in stages if st.resource.kind == "gpu"]
    gpu_cond = " || ".join(f"rank == {r}" for r in gpu_ranks) or "false"
    usage = "<image_list.txt>" if opts["mode"] == "infer" else "<image>"
    L += ["int main(int argc, char** argv)", "{",
          "    int provided;",
          "    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);",
          "    if (provided < MPI_THREAD_MULTIPLE)", "    {",
          '        fprintf(stderr, "MPI does not provide MPI_THREAD_MULTIPLE\\n");',
          "        MPI_Finalize();", "        return -1;", "    }",
          "    int rank, size;", "    MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
          "    MPI_Comm_size(MPI_COMM_WORLD, &size);",
          f"    if (argc != 2 || size != {len(stages)})", "    {",
          f'        if (rank == 0) fprintf(stderr, "Usage: mpirun -np {len(stages)} -rf rankfile %s {usage}\\n", argv[0]);',
          "        MPI_Finalize();", "        return -1;", "    }",
          "    if (rank == 0)",
          '        fprintf(stderr, "[ncnn] fp16=%d fp16_arith=%d winograd=%d sgemm=%d int8=%d packing=%d\\n", '
          "OPT_FP16, OPT_FP16_ARITH, OPT_WINOGRAD, OPT_SGEMM, OPT_INT8, OPT_PACKING);"]
    if opts["mode"] == "infer":
        L += ["    if (rank == 0)",
              '        fprintf(stderr, "[env] DUMP_BUFFER=%d ATTACK_ENABLED=%d ATTACK_TARGET_LAYER=%s FM_DIR=%s\\n", '
              'env_on("DUMP_BUFFER"), env_on("ATTACK_ENABLED"), getenv("ATTACK_TARGET_LAYER") ? '
              'getenv("ATTACK_TARGET_LAYER") : "-", getenv("FM_DIR") ? getenv("FM_DIR") : ".");']
    L += ["    if (getenv(\"IMG_OFFSET\")) g_img_offset = atoi(getenv(\"IMG_OFFSET\"));",
          f"    const bool gpu_rank = {gpu_cond};",
          "    if (gpu_rank) ncnn::create_gpu_instance();",
          "    g_blob_pool_allocator.set_size_compare_ratio(0.0f);",
          "    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);"]
    for st in stages:
        L += [f"    if (rank == {st.rank})", "    {",
              f"        model{st.rank}_init();", f"        rank{st.rank}_main(argv[1]);",
              f"        model{st.rank}_close();", "    }"]
    L += ["    if (gpu_rank) ncnn::destroy_gpu_instance();",
          "    MPI_Finalize();", "    return 0;", "}", ""]
    return "\n".join(L)
