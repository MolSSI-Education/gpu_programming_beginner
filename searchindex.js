Search.setIndex({"docnames": ["01-introduction", "02-basic-concepts", "03-cuda-program-model", "04-gpu-compilation-model", "05-cuda-execution-model", "index", "setup"], "filenames": ["01-introduction.md", "02-basic-concepts.md", "03-cuda-program-model.md", "04-gpu-compilation-model.md", "05-cuda-execution-model.md", "index.rst", "setup.md"], "titles": ["Introduction", "Basic Concepts in CUDA Programming", "CUDA Programming Model", "CUDA GPU Compilation Model", "CUDA Execution Model", "Fundamentals of Heterogeneous Parallel Programming with CUDA C/C++", "Setup"], "terms": {"question": [0, 1, 2, 3, 4, 5], "what": [0, 1, 2, 3, 4, 5], "i": [0, 1, 2, 3, 4, 5, 6], "where": [0, 2, 4, 5, 6], "did": [0, 2, 5], "come": [0, 3, 5], "from": [0, 1, 2, 3, 4, 5, 6], "how": [0, 1, 2, 3, 4, 5], "evolv": [0, 5], "ar": [0, 1, 2, 3, 4, 5, 6], "main": [0, 1, 2, 3, 4, 5], "differ": [0, 1, 2, 3, 4, 5, 6], "between": [0, 1, 2, 3, 4, 5], "cpu": [0, 1, 2, 3, 4, 5], "gpu": [0, 1, 5, 6], "architectur": [0, 2, 3, 5, 6], "relat": [0, 4, 5], "why": [0, 3, 4, 5], "do": [0, 2, 3, 5, 6], "need": [0, 1, 2, 3, 4, 5, 6], "know": [0, 2, 3, 5], "about": [0, 1, 2, 3, 4, 5, 6], "object": [0, 1, 2, 3, 4, 5], "understand": [0, 1, 2, 3, 4, 5], "fundament": [0, 4], "learn": [0, 1, 2, 3, 4, 5], "basic": [0, 3, 4, 5, 6], "aspect": [0, 1, 2, 4, 5], "softwar": [0, 4, 6], "model": [0, 1, 5], "an": [0, 1, 2, 3, 4, 5, 6], "initi": [0, 1, 2, 3, 4, 5], "high": [0, 5, 6], "perform": [0, 1, 2, 4, 5], "comput": [0, 1, 4, 5, 6], "hpc": [0, 5], "highli": 0, "multi": 0, "disciplinari": 0, "area": 0, "research": 0, "intersect": 0, "system": [0, 2, 4, 6], "hardwar": [0, 2, 4, 6], "The": [0, 1, 2, 3, 4, 5, 6], "goal": 0, "deliv": 0, "troughput": 0, "effici": [0, 3, 4, 5], "solut": 0, "computation": 0, "expens": 0, "problem": [0, 2, 3], "via": [0, 2], "simultan": [0, 1, 2, 3, 5], "us": [0, 1, 2, 4, 5, 6], "multipl": [0, 2, 3, 4, 5, 6], "process": [0, 1, 2, 3, 4, 5, 6], "unit": [0, 2, 4, 6], "invent": 0, "graphic": [0, 4, 6], "more": [0, 2, 3, 4, 5, 6], "than": [0, 1, 2], "two": [0, 1, 2, 4, 5], "decad": 0, "ago": 0, "nvidia": [0, 2, 5, 6], "wa": [0, 2, 3, 4], "follow": [0, 1, 2, 3, 4, 5, 6], "signific": [0, 4], "improv": [0, 1, 2, 4], "both": [0, 1, 2, 3, 4], "design": [0, 4, 6], "dure": [0, 3, 4, 5], "thi": [0, 1, 2, 3, 4, 5, 6], "period": 0, "ha": [0, 1, 2, 3, 4, 6], "introduc": [0, 4], "new": [0, 1, 2, 3, 4], "roughli": 0, "everi": [0, 2, 3, 4], "year": 0, "tesla": [0, 6], "2007": 0, "fermi": [0, 4, 6], "2009": 0, "kepler": [0, 5, 6], "2012": 0, "maxwel": [0, 6], "2014": 0, "pascal": [0, 4, 6], "2016": 0, "volta": [0, 6], "2017": 0, "ture": [0, 4, 5], "2018": 0, "amper": [0, 6], "2020": 0, "aforement": [0, 2, 3], "often": [0, 1, 2, 6], "part": [0, 1, 2, 3, 4, 6], "product": 0, "line": [0, 1, 2, 3, 6], "famili": 0, "tegra": 0, "mobil": 0, "embed": 0, "devic": [0, 1, 3, 4, 5, 6], "smart": [0, 4], "phone": 0, "tablet": 0, "geforc": [0, 2, 4, 5, 6], "titan": 0, "built": [0, 2], "consum": [0, 4], "orient": 0, "entertain": 0, "task": [0, 1, 2, 4, 6], "quadro": 0, "creat": [0, 3, 4], "profession": 0, "visual": [0, 6], "optim": [0, 2, 4], "technic": 0, "scientif": [0, 2], "jetson": 0, "suitabl": [0, 4], "artifici": 0, "intellig": 0, "ai": 0, "driven": [0, 2, 4], "autonom": 0, "machin": [0, 5, 6], "As": [0, 1, 2, 3, 4], "detail": [0, 2, 3, 4, 6], "specif": [0, 1, 2, 3, 4, 6], "section": [0, 1, 2, 4, 6], "we": [0, 1, 2, 3, 4, 5, 6], "enabl": [0, 1, 5, 6], "micro": [0, 4], "throughout": [0, 5, 6], "tutori": [0, 2, 3, 4, 5, 6], "programm": [0, 1, 2, 3, 4, 5], "might": [0, 2, 3, 4, 6], "see": [0, 2, 3, 4, 6], "construct": [0, 1, 2], "compris": [0, 2], "data": [0, 1, 2, 3, 4, 5], "instruct": [0, 2, 3, 4, 6], "In": [0, 1, 2, 3, 4, 6], "absenc": 0, "depend": [0, 2, 4, 6], "set": [0, 2, 3, 4, 5, 6], "e": [0, 2, 3, 4], "g": [0, 6], "result": [0, 2, 3, 4, 6], "gener": [0, 2, 3, 4], "first": [0, 2, 3, 4], "requir": [0, 2, 3, 5, 6], "anoth": [0, 2], "second": [0, 1, 2, 3, 4], "sequenti": [0, 2], "serial": [0, 4], "code": [0, 1, 2, 3, 4, 5, 6], "can": [0, 1, 2, 3, 4, 5, 6], "run": [0, 1, 2, 3, 4, 5, 6], "independ": [0, 2, 6], "concurr": [0, 1, 2, 4], "therefor": [0, 2, 3, 4, 6], "type": [0, 1, 2, 3, 4, 6], "realiz": [0, 2, 4], "each": [0, 1, 2, 3, 4, 6], "ii": [0, 1, 2, 4], "base": [0, 2, 3, 4, 6], "distribut": [0, 2, 3, 4], "mainli": 0, "becaus": [0, 1, 2, 3, 4, 6], "some": [0, 2, 3, 4, 5, 6], "function": [0, 1, 3, 4, 5], "abl": [0, 1, 2, 3, 4], "oper": [0, 2, 4, 5, 6], "meanwhil": [0, 6], "deloc": 0, "across": [0, 4], "group": [0, 2, 3], "processor": [0, 1, 3, 4], "correspond": [0, 2, 3, 4, 6], "order": [0, 1, 2, 3, 4, 6], "write": [0, 2, 4, 5, 6], "homogen": 0, "adopt": [0, 1, 2, 4, 5, 6], "which": [0, 1, 2, 3, 4, 5, 6], "one": [0, 1, 2, 3, 4, 5], "same": [0, 1, 2, 3, 4, 6], "howev": [0, 1, 2, 3, 4, 6], "offer": [0, 4, 6], "rigor": 0, "altern": 0, "respons": [0, 2], "here": [0, 1, 2, 3, 6], "intens": [0, 4], "central": [0, 4], "overal": [0, 1], "compar": [0, 2], "To": [0, 6], "better": [0, 4], "concept": [0, 2, 5], "should": [0, 1, 2, 3, 4, 5, 6], "latenc": 0, "durat": 0, "its": [0, 1, 2, 3, 4, 5, 6], "begin": [0, 2, 4], "complet": [0, 2], "express": [0, 1, 3], "microsecond": [0, 4], "m": [0, 4, 6], "": [0, 1, 2, 4, 5, 6], "throughput": [0, 4], "number": [0, 1, 2, 3, 4, 6], "per": [0, 4], "time": [0, 2, 3, 4, 5, 6], "gigaflop": 0, "flop": 0, "translat": [0, 2], "billion": 0, "float": [0, 2, 3, 4], "bandwidth": 0, "amount": [0, 1, 2], "megabyt": 0, "b": [0, 2, 3], "gigabyt": 0, "now": [0, 2, 3], "let": [0, 1, 2, 3, 4], "u": [0, 1, 2, 3, 4], "motiv": [0, 2], "behind": [0, 2, 4], "term": [0, 2], "just": [0, 1, 2], "clearli": [0, 4], "memori": [0, 1, 3, 4, 5, 6], "space": [0, 2], "assembl": [0, 3], "dynam": 0, "random": [0, 2, 3], "access": [0, 1, 2, 4, 6], "dram": 0, "lower": 0, "cach": [0, 4], "have": [0, 1, 2, 3, 4, 5], "been": [0, 1, 2, 3, 5], "static": 0, "sram": 0, "core": [0, 4, 6], "arm": [0, 5, 6], "larg": [0, 4, 6], "level": [0, 2, 4, 5], "l1": [0, 4], "l3": 0, "rare": 0, "l4": 0, "allow": [0, 2, 3, 4, 6], "them": [0, 2, 3, 4], "reduc": 0, "through": [0, 1, 2, 3, 4, 6], "benefit": 0, "principl": 0, "local": [0, 2, 3, 6], "specul": 0, "execut": [0, 2, 3, 5], "By": [0, 3], "store": [0, 2], "frequent": [0, 2], "predict": 0, "next": [0, 1, 2, 3, 4], "out": [0, 2, 3, 4], "attempt": [0, 2, 4], "minim": 0, "storag": [0, 6], "particular": [0, 2, 4], "clock": [0, 2], "speed": [0, 6], "capac": [0, 6], "size": [0, 2, 3, 4, 6], "increas": [0, 4], "chip": 0, "physic": 0, "closer": 0, "l2": 0, "onto": 0, "motherboard": 0, "modul": 0, "best": [0, 2, 4], "suit": 0, "complex": [0, 2], "logic": [0, 3, 4], "workload": 0, "short": [0, 3], "sequenc": 0, "thousand": [0, 2], "transistor": 0, "hide": 0, "maxim": [0, 4], "handl": [0, 5], "activ": [0, 2, 4, 6], "thread": [0, 1, 3, 4, 5], "when": [0, 1, 2, 3, 4], "wait": [0, 2], "fetch": [0, 2], "other": [0, 2, 4, 6], "start": [0, 1, 2, 3, 4, 5, 6], "call": [0, 1, 2, 3, 4, 6], "singl": [0, 2], "simt": [0, 2, 4], "highlight": 0, "simpl": [0, 1, 2, 4, 6], "control": [0, 1, 2, 6], "flow": [0, 1, 2, 3], "pci": 0, "bu": [0, 6], "connect": [0, 1, 6], "At": [0, 6], "gain": [0, 2, 4], "insight": [0, 4, 5], "focu": [0, 2, 4, 6], "our": [0, 2, 3, 4, 5, 6], "har": 0, "power": [0, 1, 2], "sinc": [0, 2, 3, 4, 6], "releas": [0, 4, 6], "unifi": [0, 2, 4], "becom": [0, 2, 4, 5], "major": [0, 2, 4, 6], "standard": [0, 1, 3], "purpos": [0, 4], "gpgpu": 0, "coin": [0, 2], "mark": [0, 2], "harri": [0, 2], "non": [0, 3, 4], "applic": [0, 1, 2, 4, 6], "provid": [0, 1, 2, 4, 6], "compil": [0, 1, 2, 4, 5, 6], "direct": [0, 1, 2, 3, 4, 6], "interfac": [0, 2, 4, 6], "api": [0, 2, 3, 4], "languag": [0, 1, 5, 6], "extens": [0, 2, 3, 6], "c": [0, 1, 2, 3, 4, 6], "python": 0, "fortran": 0, "etc": 0, "acceler": [0, 1, 2, 4, 6], "librari": [0, 1, 3, 6], "expos": [0, 1, 2], "hierarchi": [0, 1, 3, 5], "user": [0, 2, 4], "great": [0, 1], "over": [0, 1, 2, 4, 6], "discuss": [0, 1], "few": [0, 2], "lesson": [0, 1, 2, 4, 5], "develop": [0, 6], "environ": [0, 1, 6], "also": [0, 2, 3, 4, 6], "tool": [0, 5, 6], "includ": [0, 1, 2, 3, 4, 6], "manag": [0, 1, 3, 4, 5, 6], "nsight": [0, 4], "integr": 0, "id": [0, 4, 6], "gdb": 0, "debug": [0, 3], "command": [0, 1, 2, 3, 6], "variant": [0, 2], "profil": [0, 2, 5, 6], "analysi": [0, 4], "memcheck": 0, "launch": [1, 2, 5, 6], "abil": [1, 5], "recognit": [1, 5], "similar": [1, 2, 3, 4, 5], "semant": [1, 3, 4, 5], "those": [1, 2, 3, 4, 5, 6], "plan": 1, "heterogen": [1, 4, 6], "parallel": [1, 3, 4, 6], "work": [1, 2, 3], "co": 1, "stand": [1, 2, 6], "alon": [1, 6], "consist": [1, 2, 4, 6], "separ": [1, 2, 4], "domain": [1, 2, 3, 5], "host": [1, 2, 3, 4, 5, 6], "transfer": [1, 2, 3, 4, 5], "A": [1, 3, 4, 6], "classic": 1, "exampl": [1, 2, 3, 4], "ani": [1, 2, 3, 6], "print": [1, 2, 3, 4], "hello": [1, 2], "world": [1, 2, 4], "string": 1, "output": [1, 2, 4, 6], "stream": [1, 2, 4, 5, 6], "implement": [1, 2, 3], "hellofromcpu": 1, "hellofromgpu": 1, "name": [1, 2, 3, 4, 6], "suggest": [1, 2, 4], "messag": [1, 3, 4], "open": [1, 2, 3, 4], "file": [1, 4, 5, 6], "cu": [1, 2, 3, 4], "copi": [1, 2, 3], "stdlib": [1, 2, 3, 4], "h": [1, 2, 3, 4], "For": [1, 2, 4, 6], "statu": [1, 2, 3], "marco": [1, 3, 5], "stdio": [1, 2, 3, 4], "printf": [1, 2, 3, 4], "void": [1, 2, 3], "n": [1, 2, 3, 4, 6], "__global__": [1, 2, 3], "int": [1, 2, 3, 4], "argc": [1, 2, 3, 4], "char": [1, 2, 3, 4], "argv": [1, 2, 3, 4], "cudadevicereset": [1, 2], "hous": [1, 2], "keep": [1, 2], "return": [1, 2, 3, 4], "exit_success": [1, 2, 3, 4], "after": [1, 2, 3, 4, 6], "save": [1, 2, 3, 4], "close": 1, "folder": 1, "within": [1, 2, 3, 4, 5], "termin": [1, 2, 3, 6], "shell": [1, 2, 3, 4, 6], "nvcc": [1, 2, 4, 5], "o": [1, 2, 3, 4, 5], "If": [1, 2, 3], "your": [1, 3, 4, 6], "familiar": [1, 2, 3, 5], "gnu": [1, 3, 6], "you": [1, 2, 3, 4], "probabl": [1, 4], "notic": [1, 2], "syntax": [1, 2], "gcc": [1, 2, 6], "meain": 1, "tripl": 1, "angular": 1, "bracket": 1, "shortli": 1, "subsec": 1, "mention": [1, 2, 4], "earlier": 1, "review": 1, "analyz": [1, 2, 4], "piec": 1, "distinguish": [1, 2, 3, 5], "like": [1, 2, 3, 4, 6], "written": [1, 2, 3], "necessari": [1, 2, 3, 6], "header": [1, 3], "preprocessor": [1, 2], "macro": [1, 2, 3], "greater": [1, 2], "portabl": 1, "exit_failur": [1, 3], "show": [1, 2, 4, 6], "success": 1, "failur": 1, "format": [1, 2, 3], "describ": [1, 2], "pure": 1, "respect": [1, 2, 3, 4, 6], "definit": [1, 3], "take": [1, 2, 3, 4, 5, 6], "form": [1, 2, 3], "returntyp": 1, "functionnam": 1, "parameterlist": 1, "functionimplement": 1, "input": [1, 2, 3], "paramet": [1, 2], "all": [1, 2, 3, 4], "doe": [1, 2, 3, 5], "screen": [1, 2, 4, 6], "individu": [1, 2], "thei": [1, 6], "mani": [1, 2, 3, 4, 5], "__declarationspecification__": 1, "kernelnam": 1, "kernelimplement": 1, "declar": [1, 2, 3], "qualifi": [1, 2, 3], "indic": [1, 2, 4, 6], "callabl": 1, "capabl": [1, 5, 6], "3": [1, 4, 5], "0": [1, 2, 3, 4, 5, 6], "__device__": [1, 2], "onli": [1, 2, 3, 4, 6], "__host__": [1, 2, 3], "must": [1, 2, 3], "previou": [1, 2, 4, 5], "seen": 1, "exist": [1, 4], "Not": [1, 2], "surprisingli": 1, "grid": [1, 3, 4, 5], "block": [1, 2, 3, 4, 5], "extend": 1, "ad": [1, 2, 3], "configur": [1, 2, 4], "organ": [1, 2, 3, 4, 5], "most": [1, 2], "critic": [1, 3], "uniqu": [1, 2, 4], "mean": [1, 2], "argument": [1, 2, 3], "layout": 1, "import": [1, 2, 4, 5], "convent": 1, "latter": [1, 2, 3], "asynchron": [1, 2, 3], "back": [1, 2, 3, 4], "right": [1, 2, 6], "preliminari": 1, "measur": [2, 5], "wall": [2, 5], "logist": [2, 5], "typic": [2, 4, 5, 6], "lack": 2, "crucial": 2, "illustr": [2, 4], "final": [2, 3, 4], "emploi": 2, "befor": [2, 3, 6], "get": [2, 3], "formal": 2, "potenti": 2, "impact": [2, 4, 5], "cpuprint": 2, "defin": [2, 3, 4], "8": [2, 3, 4, 6], "nlim": 2, "idx": [2, 3], "d": [2, 3, 4], "proceed": 2, "text": [2, 3, 4], "cpu_print": 2, "Then": [2, 3], "5": [2, 3, 4, 5, 6], "6": [2, 3, 6], "7": [2, 5, 6], "fix": [2, 3, 4], "iter": 2, "statement": [2, 3], "assum": 2, "equal": [2, 3, 4], "avail": [2, 3, 4, 5, 6], "demonstr": [2, 4], "refactor": 2, "accord": [2, 3, 4], "assumpt": 2, "strategi": 2, "step": [2, 3, 4], "case": [2, 4], "total": [2, 4], "gpuprint": 2, "modifi": 2, "gpu_printer_sb": 2, "script": 2, "look": [2, 3, 4, 6], "threadidx": [2, 3], "x": [2, 3, 4, 6], "rais": 2, "error": [2, 4, 5], "1024": [2, 3, 4], "cudadevicesynchron": [2, 3, 4], "give": [2, 4], "desir": 2, "pai": 2, "attent": 2, "suppos": 2, "simplest": 2, "approach": [2, 3, 4], "well": [2, 3, 4], "until": 2, "larger": 2, "threshold": 2, "valu": 2, "limit": [2, 4], "maximum": [2, 4], "resort": 2, "while": [2, 4], "match": 2, "continu": [2, 3], "being": 2, "among": [2, 4], "cope": 2, "past": [2, 3], "empti": [2, 3], "gpu_printer_mb_loc": 2, "index": [2, 3, 4], "restart": 2, "zero": [2, 6], "go": [2, 4], "schemat": 2, "shown": [2, 4], "below": 2, "pattern": 2, "15": 2, "three": [2, 3, 4, 6], "reproduc": 2, "global": [2, 3, 4], "formula": [2, 4], "convers": 2, "tag": [2, 4], "label": [2, 4], "eq": [2, 4], "localtoglob": 2, "globalthreadidx": 2, "_q": 2, "q": [2, 4], "blockidx": [2, 3], "blockdim": [2, 3], "qquad": [2, 4], "quad": [2, 4], "y": [2, 4], "z": [2, 4], "ref": 2, "convert": 2, "gpu_printer_mb_glob": 2, "hand": [2, 4, 6], "side": [2, 3, 4, 6], "shift": 2, "offset": 2, "compens": 2, "eight": 2, "four": [2, 4], "yield": [2, 3, 4], "expect": 2, "although": [2, 4], "preserv": 2, "reason": 2, "almost": 2, "guarante": 2, "dispatch": 2, "multiprocessor": [2, 4, 5, 6], "happen": [2, 3, 4], "opposit": 2, "smaller": [2, 3], "element": 2, "target": [2, 4, 5, 6], "techniqu": 2, "present": [2, 4, 6], "hi": 2, "blog": 2, "post": 2, "deal": 2, "scenario": 2, "extra": 2, "address": 2, "alloc": [2, 3, 4], "unalloc": 2, "spot": 2, "lead": [2, 4], "incorrect": 2, "undefin": [2, 3], "behavior": 2, "avoid": [2, 4], "add": [2, 3], "numthread": 2, "condit": 2, "make": [2, 3, 4, 6], "chang": [2, 3, 4], "last": [2, 3, 4, 6], "gpu_printer_monolith": 2, "sure": [2, 6], "beyond": [2, 3], "32": [2, 4], "abov": [2, 4, 6], "It": [2, 6], "interest": [2, 3], "remind": 2, "situat": [2, 3], "less": 2, "map": 2, "increment": 2, "round": 2, "50": [2, 4], "25": 2, "24": [2, 3, 4], "26": [2, 4], "49": [2, 4], "gpu_printer_grid_stride_loop": 2, "manual": 2, "good": 2, "onc": [2, 6], "again": 2, "veri": [2, 4, 6], "small": [2, 4], "simplic": [2, 4], "clariti": 2, "hard": 2, "ideal": [2, 6], "inform": [2, 4, 6], "calcul": [2, 4], "runtim": [2, 3, 4, 5], "introduct": [2, 5], "regard": [2, 4], "figur": [2, 4], "invok": 2, "own": [2, 3, 4], "privat": [2, 4], "cooper": 2, "thank": 2, "share": [2, 4], "visibl": 2, "life": 2, "synchron": 2, "identifi": [2, 3, 4], "refer": [2, 3, 4, 6], "variabl": 2, "integ": [2, 4, 6], "vector": [2, 3, 4], "uint3": 2, "assign": [2, 4], "intern": 2, "compon": 2, "dimension": [2, 4], "dimens": 2, "bock": 2, "griddim": 2, "dim3": [2, 3, 4], "field": 2, "cartesian": 2, "cuda_runtim": [2, 3, 4], "printthreadid": 2, "numarrai": 2, "numblock": 2, "mechanist": 2, "addit": [2, 3], "load": [2, 3], "pass": 2, "style": 2, "cuda_runtime_api": 2, "upon": [2, 3, 4], "wrapper": [2, 3, 5], "conveni": [2, 3], "long": 2, "inclus": 2, "so": [2, 3, 6], "try": [2, 3, 6], "even": [2, 3], "remov": [2, 4], "still": [2, 3, 4, 6], "without": [2, 3], "issu": [2, 3, 4], "toolkit": [2, 3, 4, 5, 6], "document": [2, 3, 4, 6], "identif": 2, "previous": [2, 4, 6], "constructor": 2, "list": [2, 3, 4, 6], "automat": [2, 6], "ignor": 2, "thrdorg": 2, "left": [2, 3, 4], "frac": [2, 4], "numberofel": 2, "trigger": 2, "pre": [2, 3], "readabl": [2, 3], "backslash": [2, 3], "split": [2, 3, 4], "destroi": 2, "state": 2, "current": 2, "collect": [2, 4], "studi": 2, "formul": 2, "seri": [2, 4], "move": [2, 3], "least": 2, "dealloc": 2, "descript": [2, 3], "malloc": [2, 3, 4], "cudamalloc": [2, 3, 4], "uniniti": 2, "memset": [2, 3, 4], "cudamemset": 2, "free": [2, 3, 4], "cudafre": [2, 3, 4], "memcpi": [2, 4], "cudamemcpi": [2, 3, 4], "tabl": [2, 6], "easier": [2, 3], "lowercamelcas": 2, "java": 2, "cudaerror_t": [2, 3], "devptr": 2, "size_t": [2, 3, 4], "byte": 2, "linear": 2, "doubl": [2, 3], "pointer": 2, "togeth": 2, "except": [2, 3], "enumer": [2, 3], "With": [2, 3, 4], "signatur": [2, 3], "dst": 2, "const": [2, 3], "src": 2, "count": 2, "cudamemcpykind": 2, "kind": 2, "sourc": [2, 4, 5], "destin": 2, "infer": [2, 4], "cudamemcpyhosttohost": 2, "cudamemcpyhosttodevic": [2, 3, 4], "cudamemcpydevicetohost": [2, 3, 4], "cudamemcpydevicetodevic": 2, "cudamemcpydefault": 2, "recommend": [2, 4], "chosen": [2, 6], "scr": 2, "virtual": 2, "uva": 2, "support": [2, 3, 4, 6], "consid": [2, 4, 6], "immedi": [2, 3], "stop": 2, "renam": 2, "gpuvectorsum": [2, 3, 4], "stdbool": [2, 3], "sy": [2, 3], "inlin": [2, 3], "chronomet": [2, 3, 4], "struct": [2, 3], "timezon": [2, 3], "tzp": [2, 3], "timev": [2, 3], "tp": [2, 3], "tmp": [2, 3], "gettimeofdai": [2, 3], "tv_sec": [2, 3], "tv_usec": [2, 3], "datainiti": [2, 3, 4], "inputarrai": [2, 3], "time_t": [2, 3], "t": [2, 3], "srand": [2, 3], "unsign": [2, 3], "rand": [2, 3], "rand_max": [2, 3], "arraysumonhost": [2, 3, 4], "arraysumondevic": [2, 3, 4], "arrayequalitycheck": [2, 3, 4], "hostptr": [2, 3, 4], "deviceptr": [2, 3, 4], "toler": [2, 3], "0e": [2, 3], "bool": [2, 3], "isequ": [2, 3], "true": [2, 3], "ab": [2, 3], "fals": [2, 3], "NOT": [2, 3], "dth": [2, 3], "2f": [2, 3], "break": [2, 3], "kick": [2, 3, 4], "off": [2, 3, 4, 6], "setup": [2, 3, 4], "deviceidx": [2, 3, 4], "cudasetdevic": [2, 3, 4], "properti": [2, 3, 4], "cudadeviceprop": [2, 3], "deviceprop": [2, 3], "cudagetdeviceproperti": [2, 3, 4], "16777216": [2, 3, 4], "64": [2, 3, 4], "mb": [2, 3, 4], "vecsiz": [2, 3, 4], "vecsizeinbyt": [2, 3, 4], "sizeof": [2, 3, 4], "lu": [2, 3, 4], "h_a": [2, 3, 4], "h_b": [2, 3, 4], "tstart": [2, 3], "telaps": [2, 3], "elaps": [2, 3], "f": [2, 3], "d_a": [2, 3, 4], "d_b": [2, 3, 4], "d_c": [2, 3, 4], "numthreadsinblock": [2, 3, 4], "cudagetlasterror": [2, 3, 4], "check": [2, 3, 4, 6], "test": [2, 4], "gtx": [2, 4, 5, 6], "1650": [2, 4, 5, 6], "757348": 2, "062009": 2, "16384": 2, "001885": 2, "ten": 2, "faster": 2, "relev": 2, "practic": [2, 4], "perspect": [2, 4], "intend": [2, 6], "overst": 2, "observ": 2, "length": [2, 3], "restructur": 2, "talk": 2, "littl": 2, "bit": 2, "involv": 2, "boolean": 2, "iii": 2, "comment": [2, 3, 4], "star": 2, "shape": 2, "util": [2, 4, 6], "window": 2, "wai": [2, 4, 6], "further": [2, 3, 4, 6], "glanc": [2, 5, 6], "advanc": [2, 4], "linux": [2, 5], "book": 2, "accept": 2, "locat": 2, "fill": 2, "counterpart": 2, "specifi": [2, 6], "hundr": 2, "comparison": [2, 6], "driver": [2, 3, 4, 6], "instal": 2, "miscellan": 2, "choos": [2, 4, 5], "shit": 2, "bitwis": 2, "prepend": [2, 4], "prefix": 2, "h_": 2, "d_": 2, "sum": 2, "instead": [2, 4], "rememb": 2, "content": [2, 3, 4], "finish": 2, "htod": [2, 4], "job": [2, 3], "intermedi": [2, 4], "constantli": 2, "ey": 2, "factor": [2, 4], "find": [2, 3, 4], "possibl": [2, 3, 4], "eqref": [2, 4], "determin": 2, "tell": 2, "done": [2, 3], "resourc": [2, 4], "simd": 2, "flynn": 2, "taxonomi": 2, "wherea": 2, "dtoh": [2, 4], "stage": [2, 3, 4], "housekeep": 2, "program": [3, 4, 6], "mechan": [3, 5], "phase": [3, 5], "mode": [3, 4, 5], "master": [3, 4, 5], "poplar": 3, "llvm": 3, "infrastructur": 3, "combin": [3, 6], "kernel": [3, 4, 5], "proprietari": 3, "emb": 3, "fatbinari": 3, "imag": 3, "link": [3, 4], "procedur": 3, "exact": 3, "scope": 3, "reader": 3, "ptx": 3, "isa": 3, "summat": [3, 4], "arrai": [3, 4], "site": [3, 4], "baseurl": [3, 4], "_episod": [3, 4], "03": [3, 4], "md": [3, 4], "structur": [3, 5], "option": [3, 4, 6], "whole": 3, "11": [3, 6], "anymor": 3, "ccode": [3, 4], "guard": 3, "ifndef": 3, "ccode_h": 3, "endif": 3, "contain": 3, "tmpxft_000050f6_00000000": 3, "11_test": 3, "6_test": 3, "cudafe1": 3, "cpp": 3, "0x16a": 3, "0x181": 3, "0x227": 3, "0x477": 3, "collect2": 3, "ld": 3, "exit": 3, "linker": 3, "complain": 3, "receiv": 3, "put": 3, "guess": 3, "archiv": 3, "rearrang": 3, "symbol": 3, "usual": 3, "bug": 3, "withing": 3, "someth": 3, "els": 3, "trivial": 3, "default": [3, 4, 6], "who": 3, "experi": 3, "extern": [3, 4], "snippet": 3, "wrap": 3, "thin": 3, "around": [3, 4], "successfulli": 3, "pertin": 3, "cudacod": [3, 4], "cudacode_h": 3, "forget": 3, "either": [3, 6], "regiment": 3, "packag": [3, 6], "shorter": 3, "There": [3, 4, 6], "opportun": [3, 4], "encapsul": 3, "subsequ": 3, "ask": 3, "made": 3, "incorpor": 3, "under": 3, "deviceproperti": [3, 4], "note": 3, "natur": 3, "difficult": 3, "troubleshoot": 3, "sever": 3, "consecut": [3, 4], "fortun": 3, "errorhandl": [3, 4], "funccal": 3, "errormessag": 3, "cudageterrorstr": 3, "cudasuccess": 3, "__file__": 3, "__line__": 3, "escap": 3, "charact": 3, "otherwis": 3, "captur": 3, "human": 3, "could": 3, "directli": 3, "contradict": 3, "intent": 3, "compromis": 3, "somehow": 3, "leav": 3, "un": 3, "help": [4, 5, 6], "warp": [4, 5], "establish": [4, 5], "knowledg": [4, 5], "build": [4, 5], "profici": [4, 5], "sole": 4, "enough": 4, "achiev": 4, "todai": 4, "mitig": 4, "nevertheless": [4, 5], "advantag": 4, "blind": 4, "view": 4, "framework": 4, "bridg": 4, "sm": 4, "partit": 4, "schedul": 4, "ascend": 4, "given": 4, "warpperblock": 4, "warpsperblock": 4, "bigg": 4, "lceil": 4, "threadsperblock": 4, "warpsiz": 4, "rceil": 4, "cdot": 4, "ceil": 4, "thrdperblock": 4, "sum_q": 4, "never": 4, "valuabl": 4, "regist": 4, "affect": 4, "proper": 4, "would": [4, 6], "imper": 4, "4": [4, 6], "20": 4, "80": 4, "96": 4, "16": 4, "remain": 4, "inact": 4, "wast": 4, "path": 4, "caus": 4, "diverg": 4, "loss": 4, "disabl": 4, "took": 4, "cost": 4, "seriou": 4, "deterior": 4, "irrelev": 4, "fact": 4, "hint": 4, "nvdia": 4, "technolog": 4, "breakthrough": 4, "predecessor": 4, "essenti": 4, "somewhat": 4, "trillion": 4, "featur": [4, 6], "enhanc": 4, "textur": 4, "tensor": 4, "deep": 4, "train": 4, "up": [4, 5, 6], "500": 4, "evolutionari": 4, "real": 4, "rai": 4, "trace": 4, "rt": 4, "systemat": 4, "nvvp": 4, "timelin": 4, "nvprof": 4, "interact": 4, "deprec": 4, "futur": [4, 6], "migrat": 4, "sampl": 4, "brief": [4, 5], "tradit": 4, "toward": 4, "had": 4, "broken": [4, 6], "down": 4, "04": [4, 5], "easili": 4, "custom": 4, "variou": [4, 6], "appopt": 4, "spent": 4, "re": 4, "bash": [4, 5, 6], "5906": 4, "avg": 4, "min": 4, "max": 4, "71": 4, "57": 4, "36": 4, "369m": 4, "12": 4, "123m": 4, "075m": 4, "160m": 4, "93": 4, "669m": 4, "7770m": 4, "78": 4, "54": 4, "196": 4, "20m": 4, "65": 4, "401m": 4, "190": 4, "15u": 4, "195": [4, 6], "82m": 4, "19": 4, "84": 4, "555m": 4, "389m": 4, "210m": 4, "840m": 4, "74": 4, "8394m": 4, "591": 4, "60u": 4, "23": 4, "570": 4, "08u": 4, "101": 4, "6440u": 4, "495n": 4, "254": [4, 6], "44u": 4, "cudevicegetattribut": 4, "490": 4, "82u": 4, "163": 4, "61u": 4, "146": 4, "21u": 4, "84u": 4, "387": 4, "42u": 4, "cudevicetotalmem": 4, "110": 4, "59u": 4, "cudevicegetnam": 4, "01": [4, 6], "566u": 4, "cudalaunchkernel": 4, "00": [4, 6], "9": 4, "2890u": 4, "cudevicegetpcibusid": 4, "1680u": 4, "6390u": 4, "8790u": 4, "686n": 4, "2030u": 4, "cudevicegetcount": 4, "5500u": 4, "7750u": 4, "550n": 4, "0000u": 4, "cudeviceget": 4, "5830u": 4, "cudevicegetuuid": 4, "401n": 4, "mixtur": 4, "pid": 4, "signifi": 4, "millisecond": 4, "nanosecond": 4, "were": 4, "focus": 4, "effort": 4, "actual": 4, "column": 4, "averag": 4, "minimum": 4, "toi": 4, "conclus": 4, "reach": 4, "due": 4, "ratio": 4, "overhead": 4, "rule": 4, "algorithm": 4, "fine": 4, "event": 4, "scene": 4, "stderr": 4, "log": 4, "filenam": 4, "session": [4, 6], "method": [4, 6], "guid": 4, "panel": [4, 6], "top": [4, 6], "row": 4, "end": 4, "lifetim": 4, "sub": 4, "overlap": 4, "unguid": 4, "walk": 4, "clarifi": 4, "decid": 4, "prioriti": 4, "investig": 4, "bottom": 4, "middl": 4, "repres": 4, "movement": 4, "place": 4, "correl": 4, "pleas": 4, "cours": 5, "molecular": 5, "scienc": 5, "institut": 5, "molssi": 5, "overview": [5, 6], "beginn": 5, "encourag": 5, "student": 5, "chapter": 5, "1": 5, "2": 5, "gt": 5, "740m": 5, "18": 5, "bionic": 5, "beaver": 5, "platform": [5, 6], "v11": [5, 6], "titl": 5, "my": 5, "paradigm": 5, "briefli": 6, "cuda": 6, "latest": 6, "offici": 6, "version": 6, "center": 6, "flavor": 6, "rpm": 6, "debian": 6, "runfil": 6, "network": 6, "internet": 6, "low": 6, "disk": 6, "download": 6, "prerequisit": 6, "cleaner": 6, "updat": 6, "On": 6, "nativ": 6, "straightforward": 6, "resolv": 6, "conflict": 6, "insepar": 6, "spend": 6, "action": 6, "minimalist": 6, "trick": 6, "l": 6, "dev": 6, "nv": 6, "crw": 6, "rw": 6, "root": 6, "jan": 6, "10": 6, "09": 6, "43": 6, "nvidia0": 6, "255": 6, "nvidiactl": 6, "modeset": 6, "236": 6, "uvm": 6, "243": 6, "nvme0": 6, "brw": 6, "259": 6, "nvme0n1": 6, "nvme0n1p1": 6, "nvme0n1p2": 6, "nvme0n1p3": 6, "nvme0n1p4": 6, "card": 6, "obtain": 6, "gear": 6, "icon": 6, "corner": 6, "ubuntu": 6, "uniti": 6, "desktop": 6, "search": 6, "bar": 6, "websit": 6, "along": 6, "minor": 6, "digit": 6, "belong": 6, "older": 6, "legaci": 6, "smi": 6, "deriv": 6, "nvml": 6, "monitor": 6, "ship": 6, "displai": 6, "microsoft": 6, "simpli": 6, "455": 6, "38": 6, "persist": 6, "disp": 6, "volatil": 6, "uncorr": 6, "ecc": 6, "fan": 6, "temp": 6, "perf": 6, "pwr": 6, "usag": 6, "cap": 6, "mig": 6, "00000000": 6, "42c": 6, "p8": 6, "2w": 6, "438mib": 6, "3911mib": 6, "unnecessari": 6, "replac": 6, "ellips": 6, "referenc": 6, "page": 6, "verifi": 6, "placehold": 6, "common": 6, "newer": 6, "compat": 6, "matric": 6, "thereaft": 6, "found": 6, "oss": 6, "studio": 6, "subsystem": 6, "v10": 6, "debugg": 6, "subject": 6, "consider": 6}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"introduct": 0, "overview": [0, 1, 2, 3, 4], "1": [0, 1, 2, 3, 4, 6], "background": 0, "2": [0, 1, 2, 3, 4, 6], "parallel": [0, 2, 5], "program": [0, 1, 2, 5], "paradigm": 0, "3": [0, 2, 3, 6], "cuda": [0, 1, 2, 3, 4, 5], "A": [0, 2], "platform": 0, "heterogen": [0, 5], "kei": [0, 1, 2, 3, 4], "point": [0, 1, 2, 3, 4], "basic": [1, 2], "concept": 1, "write": 1, "our": 1, "first": 1, "note": [1, 2, 4, 6], "structur": [1, 2], "kernel": [1, 2], "execut": [1, 4], "model": [2, 3, 4], "loop": 2, "prelud": 2, "thread": 2, "hierarchi": 2, "monolith": 2, "grid": 2, "stride": 2, "devic": 2, "memori": 2, "manag": 2, "4": 2, "summat": 2, "arrai": 2, "gpu": [2, 3, 4], "header": 2, "file": [2, 3], "function": 2, "definit": 2, "compil": 3, "nvidia": [3, 4], "": 3, "separ": 3, "sourc": 3, "us": 3, "nvcc": 3, "exercis": 3, "solut": 3, "error": 3, "handl": 3, "architectur": 4, "profil": 4, "tool": 4, "command": 4, "line": 4, "visual": 4, "fundament": 5, "c": 5, "prerequisit": 5, "softwar": 5, "hardwar": 5, "specif": 5, "setup": 6, "linux": 6, "pre": 6, "instal": 6, "step": 6, "known": 6, "issu": 6, "window": 6, "wsl": 6, "user": 6, "mac": 6, "o": 6}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "sphinx.ext.intersphinx": 1, "sphinx": 57}, "alltitles": {"Introduction": [[0, "introduction"]], "Overview": [[0, null], [1, null], [2, null], [3, null], [4, null]], "1. Background": [[0, "background"]], "2. Parallel Programming Paradigms": [[0, "parallel-programming-paradigms"]], "3. CUDA: A Platform for Heterogeneous Parallel Programming": [[0, "cuda-a-platform-for-heterogeneous-parallel-programming"]], "Key Points": [[0, null], [1, null], [2, null], [3, null], [4, null]], "Basic Concepts in CUDA Programming": [[1, "basic-concepts-in-cuda-programming"]], "1. Writing Our First CUDA Program": [[1, "writing-our-first-cuda-program"]], "Note": [[1, null], [1, null], [1, null], [2, null], [2, null], [2, null], [2, null], [2, null], [2, null], [4, null], [4, null], [6, null]], "2. Structure of a CUDA Program": [[1, "structure-of-a-cuda-program"]], "2.1. Writing a CUDA Kernel": [[1, "writing-a-cuda-kernel"]], "2.2. Kernel Execution in CUDA": [[1, "kernel-execution-in-cuda"]], "CUDA Programming Model": [[2, "cuda-programming-model"]], "1. Parallelizing Loops: A Prelude to Thread Hierarchy": [[2, "parallelizing-loops-a-prelude-to-thread-hierarchy"]], "1.1. Monolithic Kernels": [[2, "monolithic-kernels"]], "1.2. Grid-Stride Loops": [[2, "grid-stride-loops"]], "2. Thread Hierarchy in CUDA": [[2, "thread-hierarchy-in-cuda"]], "3. Basics of the Device Memory Management in CUDA": [[2, "basics-of-the-device-memory-management-in-cuda"]], "4. Summation of Arrays on GPUs": [[2, "summation-of-arrays-on-gpus"]], "4.1. Header Files and Function Definitions": [[2, "header-files-and-function-definitions"]], "4.2. Structure of the Program": [[2, "structure-of-the-program"]], "CUDA GPU Compilation Model": [[3, "cuda-gpu-compilation-model"]], "1. NVIDIA\u2019s CUDA Compiler": [[3, "nvidia-s-cuda-compiler"]], "2. Compiling Separate Source Files using NVCC": [[3, "compiling-separate-source-files-using-nvcc"]], "Exercise": [[3, null], [3, null]], "Solution": [[3, null], [3, null]], "3. Error Handling": [[3, "error-handling"]], "CUDA Execution Model": [[4, "cuda-execution-model"]], "1. GPU Architecture": [[4, "gpu-architecture"]], "2. Profiling Tools": [[4, "profiling-tools"]], "2.1. Command-Line NVIDIA Profiler": [[4, "command-line-nvidia-profiler"]], "2.2. NVIDIA Visual Profiler": [[4, "nvidia-visual-profiler"]], "Fundamentals of Heterogeneous Parallel Programming with CUDA C/C++": [[5, "fundamentals-of-heterogeneous-parallel-programming-with-cuda-c-c"]], "Prerequisites": [[5, null]], "Software/Hardware Specifications": [[5, null]], "Setup": [[6, "setup"]], "1. Linux": [[6, "linux"]], "Pre-installation Steps": [[6, null]], "Known Issues": [[6, null]], "2. Windows": [[6, "windows"]], "WSL Users": [[6, null]], "3. Mac OS": [[6, "mac-os"]]}, "indexentries": {}})