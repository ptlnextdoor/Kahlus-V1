# Equation Ledger

This ledger preserves mathematical formulas, research-state rules, and discarded-but-important ideas from the NFC pivot context. Status labels are conservative: `implemented`, `experimental`, `scaffolded`, `theory only`, `archived`, `deferred`, `falsified`, or `needs evidence`.

| equation_id | equation | category | source/context | current project status | where included in final dossier | notes |
| --- | --- | --- | --- | --- | --- | --- |
| EQ-001 | `Attention(Q,K,V)=softmax(QK^T/sqrt(d))V` | operator learning | Transformer baseline | implemented baseline | this ledger, master dossier | Baseline math, not NFC novelty. |
| EQ-002 | `h_{t+1}=A(x_t)h_t+B(x_t)x_t` | operator learning | Mamba/selective SSM schematic | deferred baseline | this ledger, master dossier | Mamba is not wired as a NeuroTwin backbone. |
| EQ-003 | `F_s(x,t,omega) in R^d` | latent field definition | NFC primitive | experimental | constitution, master dossier | Core NFC primitive. |
| EQ-004 | `Y_m=O_m(F_s,A_s,U,epsilon_m)` | observation operators | NFC observation model | experimental | constitution, master dossier | Replaces direct-fusion framing. |
| EQ-005 | `p(Y_{1:M} | U,A_s)=int p(F | U,A_s) prod_m p(Y_m | F,A_s) dF` | generative probabilistic model | NFC generative statement | theory only | constitution, master dossier | Not fully implemented probabilistic inference. |
| EQ-006 | `F(t+Delta)=Phi_{theta,Delta}(F(t),U_[t-H,t],A_s)` | controlled dynamics | NFC dynamics | experimental | constitution, master dossier | Implemented only as simplified neural modules. |
| EQ-007 | `dF_t=f_theta(F_t)dt+g_theta(F_t)dU_t` | controlled dynamics | Neural CDE-style update | theory only | master dossier | Preserved as future dynamics option. |
| EQ-008 | `tau partial_t F(x,t)=-F(x,t)+int_Omega K_theta(x,x',t)sigma(F(x',t))dx'+B_theta U(t)+eta(x,t)` | neural-field PDE / integro-differential dynamics | NFC field equation | theory only | constitution, master dossier | Guides model hierarchy. |
| EQ-009 | `Z_{t+1}=Z_t+Delta t[-DZ_t+K_t sigma(Z_t)+BU_t]+xi_t` | controlled dynamics | Discretized field dynamics | experimental | constitution, master dossier | Current implementation is a simplified form. |
| EQ-010 | `K_t approx U_t V_t^T` | low-rank pair kernel | Pair-kernel approximation | implemented experimental | constitution, master dossier | Implemented as low-rank pair update. |
| EQ-011 | `M_t=softmax((U_t V_t^T)/sqrt(r)+S)` | low-rank pair kernel | Pair influence | implemented experimental | constitution, master dossier | Structural prior `S` is optional. |
| EQ-012 | `Z'_t=Z_t+M_t Z_t W` | low-rank pair kernel | Node update | implemented experimental | constitution, master dossier | Pair-Operator survives as ablation idea. |
| EQ-013 | `NPU_ij(t)=E[abs(K_ij(t)-Khat_ij(t))]` | uncertainty / calibration | Neural pair uncertainty | scaffolded | ledger, master dossier | Pair uncertainty exists only in limited form. |
| EQ-014 | `nabla_w F(i,j)=sqrt(w_ij)(F_j-F_i)` | graph calculus | Weighted graph derivative | theory only | constitution, master dossier | Future regularizer. |
| EQ-015 | `L=D-W` | graph calculus | Graph Laplacian | theory only | constitution, master dossier | Future anatomy/graph constraint. |
| EQ-016 | `R_graph(F)=sum_(i,j in E) w_ij norm(F_i-F_j)^2 = Tr(F^T L F)` | graph/temporal/spectral regularizers | Graph smoothness | theory only | constitution, master dossier | Not an implemented loss. |
| EQ-017 | `(H_HRF a)(t)=int_0^infty h(tau)a(t-tau)dtau` | fMRI HRF observation | HRF convolution | experimental | constitution, master dossier | Implemented as simplified convolution. |
| EQ-018 | `Y_fMRI(p,t)=R_p H_HRF g_theta(F(.,t))+epsilon` | fMRI HRF observation | fMRI operator | experimental | constitution, master dossier | Current operator is a simplified readout. |
| EQ-019 | `tilde U_t=sum_{ell=0}^L alpha_ell U_{t-ell}` | observation operators | Stimulus lag adapter | experimental | master dossier | Implemented as causal averaging adapter. |
| EQ-020 | `Y_EEG(t)=L_s J_theta(F_t)+epsilon_EEG` | EEG/MEG observation | EEG operator | experimental | constitution, master dossier | Current operator is simplified pooling/readout. |
| EQ-021 | `Y_MEG(t)=M_s J_theta(F_t)+epsilon_MEG` | EEG/MEG observation | MEG operator | theory only | constitution, master dossier | No MEG support claim. |
| EQ-022 | `Y_{n,t} ~ Poisson(Delta t softplus(w_n^T F(x_n,t)))` | spike/calcium observation | Spike observation | theory only | constitution, master dossier | Not implemented. |
| EQ-023 | `c_{n,t}=(k_Ca*r_n)(t)+epsilon` | spike/calcium observation | Calcium observation | theory only | constitution, master dossier | Not implemented. |
| EQ-024 | `p(a_t | F_t,U_t)=softmax(C pool(F_t)+D U_t)` | behavior observation | Behavior observation | scaffolded | constitution, master dossier | Behavior readout exists, claim does not. |
| EQ-025 | `q_phi(F_{0:T} | Y_obs,U,A_s)` | variational inference / ELBO | Variational posterior | theory only | ledger, master dossier | Not implemented. |
| EQ-026 | `L_ELBO=E_q[sum_{m,t} log p_theta(Y_{m,t}|F_t,A_s)] - beta KL(q_phi(F|Y,U,A_s) || p_theta(F|U,A_s))` | variational inference / ELBO | ELBO | theory only | ledger, master dossier | Future probabilistic training path. |
| EQ-027 | `F'=AF` | gauge symmetry / identifiability | Gauge ambiguity | theory only | constitution, master dossier | Explains latent non-identifiability. |
| EQ-028 | `O'_m=O_m A^{-1}` | gauge symmetry / identifiability | Gauge ambiguity | theory only | constitution, master dossier | Must not overclaim latent semantics. |
| EQ-029 | `O'_m(F')=O_m(F)` | gauge symmetry / identifiability | Gauge invariance | theory only | constitution, master dossier | Observation equivalence. |
| EQ-030 | `R_time=sum_t norm(F_{t+1}-F_t)^2` | graph/temporal/spectral regularizers | Temporal smoothness | theory only | ledger, master dossier | Not implemented as loss. |
| EQ-031 | `R_band=sum_omega gamma_omega norm(Fhat(omega))^2` | graph/temporal/spectral regularizers | Spectral regularizer | theory only | ledger, master dossier | Not implemented. |
| EQ-032 | `A_s=I+U_s V_s^T` | operator learning | Low-rank subject adapter | scaffolded | ledger, master dossier | Subject adaptation exists elsewhere, not NFC proof. |
| EQ-033 | `dF/dt=f_theta(F,U,t)` | stability / spectral radius | Continuous stability setup | theory only | ledger, master dossier | Future stability audit. |
| EQ-034 | `J_t=partial f_theta / partial F` | stability / spectral radius | Jacobian | theory only | ledger, master dossier | Not computed. |
| EQ-035 | `Re(lambda_i(J_t))<0` | stability / spectral radius | Local stability | theory only | ledger, master dossier | Not enforced. |
| EQ-036 | `rho(A)<1` | stability / spectral radius | Discrete stability | theory only | ledger, master dossier | Not enforced. |
| EQ-037 | `R_stab=max(0,rho(A)-1)^2` | stability / spectral radius | Stability penalty | theory only | ledger, master dossier | Future loss term. |
| EQ-038 | `L_NLL=sum_{m,t,i} (y-mu)^2/(2 sigma^2)+0.5 log sigma^2` | uncertainty / calibration | Uncertainty NLL | theory only | constitution, master dossier | Current uncertainty calibration is not full NLL. |
| EQ-039 | `P[Y in C_alpha(X)] approx 1-alpha` | uncertainty / calibration | Calibration condition | theory only | constitution, master dossier | Requires real calibration artifacts. |
| EQ-040 | `L_InfoNCE=-log exp(sim(z_a,z_b)/tau)/sum_{b'} exp(sim(z_a,z_b')/tau)` | information theory / contrastive learning | Contrastive alignment | theory only | ledger, master dossier | Future representation audit. |
| EQ-041 | `g(F_{t+1})=K g(F_t)` | Koopman / residual dynamics | Koopman form | theory only | ledger, master dossier | Future dynamics baseline. |
| EQ-042 | `F_{t+1}=K F_t+R_theta(F_t,U_t)` | Koopman / residual dynamics | Koopman residual update | theory only | ledger, master dossier | Not implemented. |
| EQ-043 | `L=L_pred+lambda_R norm(R_theta)^2+lambda_K max(0,rho(K)-1)^2` | Koopman / residual dynamics | Koopman residual loss | theory only | ledger, master dossier | Not implemented. |
| EQ-044 | `C=(1/T)XX^T` | Riemannian EEG covariance | EEG covariance | theory only | ledger, master dossier | Future EEG baseline. |
| EQ-045 | `Chat=LL^T+epsilon I` | Riemannian EEG covariance | SPD covariance head | theory only | ledger, master dossier | Not implemented. |
| EQ-046 | `d_AIRM(C1,C2)=norm(log(C1^{-1/2}C2C1^{-1/2}))_F` | Riemannian EEG covariance | AIRM distance | theory only | ledger, master dossier | Not implemented. |
| EQ-047 | `T_m={k Delta_m:k in Z}` | sampling lattice / Fourier / coprime windows | Sampling grid | theory only | ledger, master dossier | Future sampling stress test. |
| EQ-048 | `t in union_m T_m` | sampling lattice / Fourier / coprime windows | Union of sampling grids | theory only | ledger, master dossier | Future multimodal timing model. |
| EQ-049 | `phi(t)=[sin(2 pi k t/T), cos(2 pi k t/T)]_{k=1}^K` | sampling lattice / Fourier / coprime windows | Fourier features | theory only | ledger, master dossier | Not implemented. |
| EQ-050 | `L v_k=lambda_k v_k` | sampling lattice / Fourier / coprime windows | Graph spectral modes | theory only | ledger, master dossier | Not implemented. |
| EQ-051 | `gcd(w,s)=1` | sampling lattice / Fourier / coprime windows | Coprime window stress test | theory only | ledger, master dossier | Future split/window audit. |
| EQ-052 | `log(phi_1(t;theta)/phi_0) approx J delta mu_a(t;theta)` | fNIRS optical/Rytov observation | Rytov approximation | theory only | fNIRS notes, master dossier | fNIRS is docs only. |
| EQ-053 | `phi_1(t;theta) approx phi_0 exp(J delta mu_a(t;theta))` | fNIRS optical/Rytov observation | Predicted intensity | theory only | fNIRS notes, master dossier | No fNIRS implementation claim. |
| EQ-054 | `Delta OD(t;theta)=log phi_base_bar - log phi_1(t;theta)` | fNIRS optical/Rytov observation | Optical density | theory only | fNIRS notes, master dossier | Docs only. |
| EQ-055 | `Y_fNIRS(t)=Delta OD(phi_0 exp(J delta mu_a(t)))+epsilon` | fNIRS optical/Rytov observation | fNIRS observation | theory only | fNIRS notes, master dossier | Docs only. |
| EQ-056 | `F_hemo(t)=(h_HRF*g(F_neural))(t)` | fNIRS optical/Rytov observation | Neural to hemodynamic field | theory only | fNIRS notes, master dossier | Informs future generator realism. |
| EQ-057 | `Y_fMRI(t)=R F_hemo(t)+epsilon` | fMRI HRF observation | Synthetic fMRI | experimental | ledger, master dossier | Current synthetic generator approximates this. |
| EQ-058 | `Y_EEG(t)=L F_neural(t)+A_motion(t)+epsilon` | EEG/MEG observation | Synthetic EEG | experimental | ledger, master dossier | Motion artifacts are future realism. |
| EQ-059 | `Q:R^d -> {0,1}^{bd}` | TurboQuant/TurboVec math | TurboQuant map | deferred | TurboQuant notes, master dossier | Optional infrastructure only. |
| EQ-060 | `E norm(x-xhat)_2^2` | TurboQuant/TurboVec math | MSE objective | deferred | TurboQuant notes, master dossier | Not implemented. |
| EQ-061 | `E[(<x,q>-<xhat,q>)^2]` | TurboQuant/TurboVec math | Inner-product objective | deferred | TurboQuant notes, master dossier | Not implemented. |
| EQ-062 | `r=norm(v)_2, u=v/norm(v)_2` | TurboQuant/TurboVec math | Vector normalization | deferred | TurboQuant notes, master dossier | Optional future adapter. |
| EQ-063 | `z=Ru` | TurboQuant/TurboVec math | Random rotation | deferred | TurboQuant notes, master dossier | Optional future adapter. |
| EQ-064 | `norm(z)_2=norm(u)_2=1` | TurboQuant/TurboVec math | Norm preservation | deferred | TurboQuant notes, master dossier | Optional future adapter. |
| EQ-065 | `z_i approx N(0,1/d)` | TurboQuant/TurboVec math | Coordinate approximation | deferred | TurboQuant notes, master dossier | Risky at low dimension. |
| EQ-066 | `q_i=c_argmin_k abs(z_i-c_k)` | TurboQuant/TurboVec math | Scalar quantization | deferred | TurboQuant notes, master dossier | Optional future adapter. |
| EQ-067 | `x=xhat_MSE+r_residual` | TurboQuant/TurboVec math | Residual split | deferred | TurboQuant notes, master dossier | Optional future adapter. |
| EQ-068 | `xhat_prod=xhat_MSE+QJL(r_residual)` | TurboQuant/TurboVec math | QJL residual correction | deferred | TurboQuant notes, master dossier | Optional future adapter. |
| EQ-069 | `score(q,x)=<q,x>` | retrieval-kNN baseline | Inner-product score | deferred | TurboQuant notes, master dossier | Use exact numpy first. |
| EQ-070 | `I_stim={Q(e_stim,i)}_{i=1}^N` | semantic duplicate audit | Stimulus index | deferred | TurboQuant notes, master dossier | Future semantic audit. |
| EQ-071 | `z_t=pool(F_t)` | latent field memory | Latent summary | deferred | TurboQuant notes, master dossier | Future memory layer. |
| EQ-072 | `z_{t,k}=pool_{i in network k}(F_{t,i})` | latent field memory | Network summary | deferred | TurboQuant notes, master dossier | Future audit feature. |
| EQ-073 | `N_k(q)={i_1,...,i_k}` | retrieval-kNN baseline | Retrieval neighborhood | deferred | TurboQuant notes, master dossier | Optional baseline. |
| EQ-074 | `yhat_test=sum_{i in N_k(q)} w_i y_i` | retrieval-kNN baseline | Retrieval-kNN prediction | deferred | TurboQuant notes, master dossier | Must not use test targets. |
| EQ-075 | `w_i=exp(tau <q,z_i>)/sum_{j in N_k(q)} exp(tau <q,z_j>)` | retrieval-kNN baseline | Retrieval weights | deferred | TurboQuant notes, master dossier | Train-only labels only. |
| EQ-076 | `abs(<q,z_i>-<q,zhat_i>) <= norm(q)_2 norm(z_i-zhat_i)_2` | TurboQuant/TurboVec math | Score error bound | deferred | TurboQuant notes, master dossier | Quantization risk. |
| EQ-077 | `d_min(x_test,D_train)=min_{x_i in D_train} norm(q(x_test)-q(x_i))` | semantic duplicate audit | Nearest train distance | deferred | TurboQuant notes, master dossier | Future leakage audit. |
| EQ-078 | `L=sum_m lambda_m L_m+lambda_latent L_latent+lambda_graph Tr(F^T L F)+lambda_stab R_stab+lambda_unc L_cal` | falsification gates | Full NFC loss | theory only | master dossier | Not implemented as one loss. |
| EQ-079 | `L_fMRI=norm(M(Yhat_fMRI-Y_fMRI))^2` | fMRI HRF observation | fMRI reconstruction loss | experimental | master dossier | MSE implemented, mask semantics vary. |
| EQ-080 | `L_EEG=norm(M(Yhat_EEG-Y_EEG))^2+gamma norm(PSD(Yhat)-PSD(Y))^2` | EEG/MEG observation | EEG loss | theory only | master dossier | PSD term not implemented. |
| EQ-081 | `L_spike=-sum_{n,t} log Poisson(y_{n,t};lambda_{n,t})` | spike/calcium observation | Spike loss | theory only | master dossier | Not implemented. |
| EQ-082 | `L_{a->b}=norm(O_b(I_theta(y_a))-y_b)` | falsification gates | Cross-modal consistency | theory only | master dossier | Field-mediated route under test. |
| EQ-083 | `T_{a->b}=O_b circ I_a` | falsification gates | Commutative translation factorization | theory only | master dossier | Research framing. |
| EQ-084 | `y_a -> y_b` | falsification gates | Direct translation baseline | implemented baseline | master dossier | Baseline lane. |
| EQ-085 | `y_a -> F -> y_b` | falsification gates | Field-mediated translation | experimental | master dossier | NFC lane. |

## Status Rules

- Pair-Operator is archived as the main primitive and retained as an ablation.
- NFC is the experimental architecture path.
- fNIRS is theory/evidence only and not implemented support.
- TurboVec/TurboQuant is optional retrieval/audit infrastructure, not the model contribution.
- MOABB Track A is a reproducibility evidence path.
- Algonauts/CNeuroMod fMRI is the future Track B model battlefield.
- MDD or clinical claims are not allowed.
- A100 experiment rules require strict 1x synthetic diagnostic before Algonauts or 6x DDP.
