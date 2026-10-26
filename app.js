const data = window.FORENSIC_AGE_MODELS;

const MODEL_ORDER = ["ols", "polynomial", "randomForest", "svr", "xgboost", "adaboost"];
const BASIS_ORDER = ["sexSpecific", "sexCombined"];
// Set demirjian to true to restore Method 1B in the UI.
const METHOD_VISIBILITY = {
  method1a: true,
  demirjian: false,
  kohler: true,
};
// Set to true if you want method-code subtitles such as "Method 1A" and "Method 2" shown again.
const SHOW_METHOD_CODE_SUBTITLES = false;

const state = {
  sex: "F",
  method: "method1a",
  selectedBases: new Set(BASIS_ORDER),
  selectedModels: new Set(MODEL_ORDER),
  hasEstimated: false,
};

const els = {
  sexOptions: document.querySelector("#sex-options"),
  methodOptions: document.querySelector("#method-options"),
  basisOptions: document.querySelector("#basis-options"),
  basisCount: document.querySelector("#basis-count"),
  featureInputs: document.querySelector("#feature-inputs"),
  measurementNote: document.querySelector("#measurement-note"),
  modelOptions: document.querySelector("#model-options"),
  modelCount: document.querySelector("#model-count"),
  estimateButton: document.querySelector("#estimate-button"),
  resetButton: document.querySelector("#reset-button"),
  selectAllButton: document.querySelector("#select-all-button"),
  clearModelsButton: document.querySelector("#clear-models-button"),
  statusMessage: document.querySelector("#status-message"),
  summaryBand: document.querySelector("#summary-band"),
  resultsList: document.querySelector("#results-list"),
  detailsStrip: document.querySelector("#details-strip"),
  trainedPill: document.querySelector("#trained-pill"),
};

function formatAge(value) {
  return `${value.toFixed(2)} years`;
}

function formatMetric(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "n/a";
}

function dot(a, b) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function predictLinear(model, vector) {
  return model.intercept + dot(model.coefficients, vector);
}

function predictPolynomial(model, vector) {
  const scaled = vector.map((value, index) => (value - model.mean[index]) / model.scale[index]);
  const terms = model.powers.map((powers) =>
    powers.reduce((product, power, index) => product * Math.pow(scaled[index], power), 1),
  );
  return model.intercept + dot(model.coefficients, terms);
}

function predictTree(tree, vector) {
  let node = 0;
  while (tree.childrenLeft[node] !== -1) {
    const featureIndex = tree.feature[node];
    const threshold = tree.threshold[node];
    node = vector[featureIndex] <= threshold ? tree.childrenLeft[node] : tree.childrenRight[node];
  }
  return tree.value[node];
}

function predictRandomForest(model, vector) {
  const total = model.trees.reduce((sum, tree) => sum + predictTree(tree, vector), 0);
  return total / model.trees.length;
}

function predictSvr(model, vector) {
  const scaled = vector.map((value, index) => (value - model.mean[index]) / model.scale[index]);
  const total = model.supportVectors.reduce((sum, supportVector, index) => {
    const squaredDistance = supportVector.reduce((distance, value, featureIndex) => {
      const diff = scaled[featureIndex] - value;
      return distance + diff * diff;
    }, 0);
    return sum + model.dualCoef[index] * Math.exp(-model.gamma * squaredDistance);
  }, 0);
  return model.intercept + total;
}

function findXgbChild(node, nodeId) {
  return node.children.find((child) => child.nodeid === nodeId);
}

function splitIndex(splitName) {
  if (typeof splitName === "number") {
    return splitName;
  }
  const text = String(splitName);
  return Number(text.startsWith("f") ? text.slice(1) : text);
}

function predictXgbTree(node, vector) {
  if (Object.prototype.hasOwnProperty.call(node, "leaf")) {
    return node.leaf;
  }
  const value = vector[splitIndex(node.split)];
  const nextId = value < node.split_condition ? node.yes : node.no;
  return predictXgbTree(findXgbChild(node, nextId), vector);
}

function predictXgboost(model, vector) {
  return model.baseScore + model.trees.reduce((sum, tree) => sum + predictXgbTree(tree, vector), 0);
}

function predictAdaboost(model, vector) {
  const weighted = model.trees
    .map((tree, index) => ({
      value: predictTree(tree, vector),
      weight: model.weights[index],
    }))
    .sort((a, b) => a.value - b.value);
  const halfWeight = model.weights.reduce((sum, weight) => sum + weight, 0) / 2;
  let runningWeight = 0;
  for (const item of weighted) {
    runningWeight += item.weight;
    if (runningWeight >= halfWeight) {
      return item.value;
    }
  }
  return weighted.at(-1).value;
}

function predictModel(model, vector) {
  switch (model.type) {
    case "linear":
      return predictLinear(model, vector);
    case "polynomial":
      return predictPolynomial(model, vector);
    case "randomForest":
      return predictRandomForest(model, vector);
    case "svr":
      return predictSvr(model, vector);
    case "xgboost":
      return predictXgboost(model, vector);
    case "adaboost":
      return predictAdaboost(model, vector);
    default:
      throw new Error(`Unsupported model type: ${model.type}`);
  }
}

function methodMeta() {
  return data.methods[state.method];
}

function visibleMethodKeys() {
  return Object.keys(data.methods).filter((methodKey) => METHOD_VISIBILITY[methodKey] !== false);
}

function selectedBasisKeys() {
  return BASIS_ORDER.filter((basisKey) => state.selectedBases.has(basisKey));
}

function getTrainedForBasis(basisKey) {
  const kohlerSelection = currentKohlerSelection();
  if (basisKey === "sexSpecific") {
    const trained = data.trained.sexSpecific[state.sex][state.method];
    return kohlerSelection.isUsable ? trained.variants[kohlerSelection.key] : trained;
  }
  const trained = data.trained.sexCombined[state.method];
  return kohlerSelection.isUsable ? trained.variants[kohlerSelection.key] : trained;
}

function vectorForBasis(basisKey, input) {
  const measurements = input.values;
  if (basisKey === "sexCombined") {
    return [data.sexEncoding[state.sex], ...measurements];
  }
  return measurements;
}

function basisShortLabel(basisKey) {
  return basisKey === "sexSpecific" ? "Specific" : "Combined";
}

function isKohlerMethod() {
  return state.method === "kohler";
}

function kohlerVariantKey(features) {
  return features.join("_");
}

function toothList(features) {
  if (!features || features.length === 0) {
    return "";
  }
  if (features.length === 1) {
    return features[0];
  }
  return `${features.slice(0, -1).join(", ")} and ${features.at(-1)}`;
}

function currentKohlerSelection() {
  if (!isKohlerMethod()) {
    return { features: [], values: [], isUsable: false, key: "" };
  }
  const method = methodMeta();
  const features = [];
  const values = [];
  method.features.forEach((feature) => {
    const control = document.querySelector(`[data-feature="${feature}"]`);
    if (!control || control.value === "") {
      return;
    }
    features.push(feature);
    values.push(Number(control.value));
  });
  return {
    features,
    values,
    isUsable: features.length >= method.minimumRequiredFeatures,
    key: kohlerVariantKey(features),
  };
}

function handleFeatureChange() {
  if (isKohlerMethod()) {
    renderBasisOptions();
    renderModelOptions();
    renderDetails();
  }
  rerunIfReady();
}

function clearNode(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function makeButton(text, className, attributes = {}) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  button.textContent = text;
  Object.entries(attributes).forEach(([key, value]) => button.setAttribute(key, value));
  return button;
}

function renderSexOptions() {
  clearNode(els.sexOptions);
  Object.entries(data.sexes).forEach(([sexKey, sex]) => {
    const button = makeButton(sex.label, "", {
      role: "radio",
      "aria-checked": String(state.sex === sexKey),
    });
    button.addEventListener("click", () => {
      state.sex = sexKey;
      renderAll();
      rerunIfReady();
    });
    els.sexOptions.appendChild(button);
  });
}

function renderMethodOptions() {
  clearNode(els.methodOptions);
  visibleMethodKeys().forEach((methodKey) => {
    const method = data.methods[methodKey];
    const button = makeButton("", "method-option", {
      role: "radio",
      "aria-checked": String(state.method === methodKey),
    });
    const title = document.createElement("strong");
    title.textContent = method.shortLabel;
    button.append(title);
    if (SHOW_METHOD_CODE_SUBTITLES) {
      const subtitle = document.createElement("span");
      subtitle.textContent = method.label;
      button.appendChild(subtitle);
    }
    button.addEventListener("click", () => {
      state.method = methodKey;
      state.hasEstimated = false;
      renderAll();
    });
    els.methodOptions.appendChild(button);
  });
}

function renderBasisOptions() {
  clearNode(els.basisOptions);
  BASIS_ORDER.forEach((basisKey) => {
    const basis = data.modelBases[basisKey];
    const trained = getTrainedForBasis(basisKey);
    const label = document.createElement("label");
    label.className = `basis-option${state.selectedBases.has(basisKey) ? " is-selected" : ""}`;

    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = state.selectedBases.has(basisKey);
    input.addEventListener("change", () => {
      if (input.checked) {
        state.selectedBases.add(basisKey);
      } else {
        state.selectedBases.delete(basisKey);
      }
      renderBasisOptions();
      renderModelOptions();
      renderDetails();
      rerunIfReady();
    });

    const text = document.createElement("span");
    const name = document.createElement("span");
    name.className = "basis-name";
    name.textContent = basis.label;
    const meta = document.createElement("span");
    meta.className = "basis-meta";
    meta.textContent =
      isKohlerMethod() && !currentKohlerSelection().isUsable
        ? "Requires at least 2 teeth"
        : `${trained.trainRows} train | ${trained.testRows} test`;
    text.append(name, meta);
    label.append(input, text);
    els.basisOptions.appendChild(label);
  });
  els.basisCount.textContent = `${state.selectedBases.size} selected`;
}

function renderFeatureInputs() {
  const method = methodMeta();
  clearNode(els.featureInputs);
  if (state.method === "method1a") {
    els.measurementNote.textContent = "Continuous value, 0 to 100";
  } else if (isKohlerMethod()) {
    els.measurementNote.textContent = `Enter at least ${method.minimumRequiredFeatures} of 4 teeth; leave unavailable teeth blank.`;
  } else {
    els.measurementNote.textContent = "Ordinal stages";
  }

  method.features.forEach((feature) => {
    const range = method.ranges[feature];
    const wrapper = document.createElement("div");
    wrapper.className = "input-field";

    const label = document.createElement("label");
    label.htmlFor = `feature-${feature}`;
    label.textContent = method.featureLabels[feature];

    if (range.step === 1) {
      const select = document.createElement("select");
      select.id = `feature-${feature}`;
      select.dataset.feature = feature;

      const empty = document.createElement("option");
      empty.value = "";
      empty.textContent = isKohlerMethod() ? "Unavailable" : "Select";
      select.appendChild(empty);

      for (let value = range.min; value <= range.max; value += 1) {
        const option = document.createElement("option");
        option.value = String(value);
        option.textContent = String(value);
        select.appendChild(option);
      }
      select.addEventListener("change", handleFeatureChange);
      wrapper.append(label, select);
    } else {
      const input = document.createElement("input");
      input.id = `feature-${feature}`;
      input.dataset.feature = feature;
      input.type = "number";
      input.min = String(range.min);
      input.max = String(range.max);
      input.step = String(range.step);
      input.inputMode = "decimal";
      input.placeholder = `${range.min} to ${range.max}`;
      input.addEventListener("input", handleFeatureChange);
      wrapper.append(label, input);
    }

    els.featureInputs.appendChild(wrapper);
  });
}

function modelMetricText(modelKey) {
  const bases = selectedBasisKeys();
  if (bases.length === 0) {
    return "Select training basis";
  }
  if (isKohlerMethod() && !currentKohlerSelection().isUsable) {
    return "Enter at least 2 Kohler teeth";
  }
  return bases
    .map((basisKey) => {
      const metric = getTrainedForBasis(basisKey).metrics[modelKey];
      return `${basisShortLabel(basisKey)} MAE ${formatMetric(metric.mae)}`;
    })
    .join(" | ");
}

function renderModelOptions() {
  clearNode(els.modelOptions);
  MODEL_ORDER.forEach((modelKey) => {
    const label = document.createElement("label");
    label.className = `model-option${state.selectedModels.has(modelKey) ? " is-selected" : ""}`;

    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = state.selectedModels.has(modelKey);
    input.addEventListener("change", () => {
      if (input.checked) {
        state.selectedModels.add(modelKey);
      } else {
        state.selectedModels.delete(modelKey);
      }
      renderModelOptions();
      rerunIfReady();
    });

    const text = document.createElement("span");
    const name = document.createElement("span");
    name.className = "model-name";
    name.textContent = data.modelLabels[modelKey];
    const meta = document.createElement("span");
    meta.className = "model-meta";
    meta.textContent = modelMetricText(modelKey);
    text.append(name, meta);
    label.append(input, text);
    els.modelOptions.appendChild(label);
  });
  els.modelCount.textContent = `${state.selectedModels.size} selected`;
}

function renderDetails() {
  const method = methodMeta();
  const basisKeys = selectedBasisKeys();
  const kohlerSelection = currentKohlerSelection();
  els.trainedPill.textContent =
    isKohlerMethod() && kohlerSelection.isUsable
      ? `Teeth ${kohlerSelection.features.join(", ")}`
      : basisKeys.length === 0
      ? "Select basis"
      : basisKeys.map((basisKey) => basisShortLabel(basisKey)).join(" + ");
  clearNode(els.detailsStrip);

  [
    ["Sex", data.sexes[state.sex].label],
    ["Method", method.shortLabel],
    ["Basis", basisKeys.length ? basisKeys.map((basisKey) => basisShortLabel(basisKey)).join(" + ") : "None"],
    ...(isKohlerMethod()
      ? [["Kohler Teeth", kohlerSelection.isUsable ? toothList(kohlerSelection.features) : "Need 2+"]]
      : []),
    ["Target", data.target],
  ].forEach(([label, value]) => {
    const item = document.createElement("div");
    item.className = "detail-item";
    const small = document.createElement("span");
    small.textContent = label;
    const strong = document.createElement("strong");
    strong.textContent = value;
    item.append(small, strong);
    els.detailsStrip.appendChild(item);
  });
}

function resetResults() {
  els.statusMessage.textContent = "";
  els.summaryBand.innerHTML = `
    <span class="summary-label">Ready</span>
    <strong>Enter measurements and run the selected models.</strong>
  `;
  clearNode(els.resultsList);
}

function readVector() {
  const method = methodMeta();
  if (isKohlerMethod()) {
    const selection = currentKohlerSelection();
    for (const feature of selection.features) {
      const value = selection.values[selection.features.indexOf(feature)];
      const range = method.ranges[feature];
      if (!Number.isFinite(value) || value < range.min || value > range.max) {
        return {
          error: `${method.featureLabels[feature]} must be between ${range.min} and ${range.max}.`,
        };
      }
    }
    if (!selection.isUsable) {
      return { error: `Enter at least ${method.minimumRequiredFeatures} Kohler teeth.` };
    }
    if (state.selectedBases.size === 0) {
      return { error: "Select at least one training basis." };
    }
    if (state.selectedModels.size === 0) {
      return { error: "Select at least one model." };
    }
    return { values: selection.values, features: selection.features, variantKey: selection.key };
  }

  const values = [];
  for (const feature of method.features) {
    const control = document.querySelector(`[data-feature="${feature}"]`);
    const raw = control.value;
    const range = method.ranges[feature];
    if (raw === "") {
      return { error: `Enter ${method.featureLabels[feature]}.` };
    }
    const value = Number(raw);
    if (!Number.isFinite(value) || value < range.min || value > range.max) {
      return {
        error: `${method.featureLabels[feature]} must be between ${range.min} and ${range.max}.`,
      };
    }
    values.push(value);
  }
  if (state.selectedBases.size === 0) {
    return { error: "Select at least one training basis." };
  }
  if (state.selectedModels.size === 0) {
    return { error: "Select at least one model." };
  }
  return { values, features: method.features };
}

function estimate() {
  const input = readVector();
  if (input.error) {
    els.statusMessage.textContent = input.error;
    return;
  }

  els.statusMessage.textContent = "";
  state.hasEstimated = true;

  const predictions = [];
  selectedBasisKeys().forEach((basisKey) => {
    const trained = getTrainedForBasis(basisKey);
    const vector = vectorForBasis(basisKey, input);
    [...state.selectedModels].forEach((modelKey) => {
      const model = trained.models[modelKey];
      predictions.push({
        basisKey,
        modelKey,
        basisLabel: data.modelBases[basisKey].label,
        modelLabel: data.modelLabels[modelKey],
        value: predictModel(model, vector),
        metric: trained.metrics[modelKey],
        features: input.features,
      });
    });
  });

  const mean =
    predictions.reduce((sum, prediction) => sum + prediction.value, 0) / predictions.length;
  const min = Math.min(...predictions.map((prediction) => prediction.value));
  const max = Math.max(...predictions.map((prediction) => prediction.value));

  els.summaryBand.innerHTML = `
    <span class="summary-label">${predictions.length} estimate${predictions.length > 1 ? "s" : ""}</span>
    <strong>${formatAge(mean)}</strong>
    <span class="range-note">${
      isKohlerMethod()
        ? `Kohler based on teeth ${toothList(input.features)}. `
        : ""
    }Selected estimate range: ${formatAge(min)} to ${formatAge(max)}</span>
  `;

  clearNode(els.resultsList);
  predictions.forEach((prediction) => {
    const card = document.createElement("article");
    card.className = "result-card";
    const text = document.createElement("div");
    const title = document.createElement("h3");
    title.textContent = prediction.modelLabel;
    const meta = document.createElement("p");
    meta.textContent = `${prediction.basisLabel}${
      isKohlerMethod() ? ` | Teeth ${prediction.features.join(", ")}` : ""
    } | Held-out test MAE ${formatMetric(
      prediction.metric.mae,
    )}, R2 ${formatMetric(prediction.metric.r2)}`;
    const value = document.createElement("div");
    value.className = "age-value";
    value.textContent = formatAge(prediction.value);
    text.append(title, meta);
    card.append(text, value);
    els.resultsList.appendChild(card);
  });
}

function rerunIfReady() {
  if (state.hasEstimated) {
    estimate();
  }
}

function renderAll() {
  if (!visibleMethodKeys().includes(state.method)) {
    state.method = visibleMethodKeys()[0];
  }
  renderSexOptions();
  renderMethodOptions();
  renderBasisOptions();
  renderFeatureInputs();
  renderModelOptions();
  renderDetails();
  resetResults();
}

els.estimateButton.addEventListener("click", estimate);

els.resetButton.addEventListener("click", () => {
  state.sex = "F";
  state.method = "method1a";
  state.selectedBases = new Set(BASIS_ORDER);
  state.selectedModels = new Set(MODEL_ORDER);
  state.hasEstimated = false;
  renderAll();
});

els.selectAllButton.addEventListener("click", () => {
  state.selectedModels = new Set(MODEL_ORDER);
  renderModelOptions();
  rerunIfReady();
});

els.clearModelsButton.addEventListener("click", () => {
  state.selectedModels.clear();
  renderModelOptions();
  rerunIfReady();
});

renderAll();
