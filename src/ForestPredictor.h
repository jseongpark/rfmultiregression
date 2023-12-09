/*-------------------------------------------------------------------------------
  This file is part of generalized random forest (grf).

  grf is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  grf is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with grf. If not, see <http://www.gnu.org/licenses/>.
 #-------------------------------------------------------------------------------*/

#ifndef GRF_FORESTPREDICTOR_H
#define GRF_FORESTPREDICTOR_H

#include "RelabelingStrategy.h"
#include "SplittingRule.h"
#include "Prediction.h"
#include "TreeTraverser.h"
#include "PredictionCollector.h"
#include "SampleWeightComputer.h"
#include "OptimizedPredictionStrategy.h"
#include "DefaultPredictionStrategy.h"
#include "Tree.h"
#include "TreeTrainer.h"
#include "Forest.h"
#include <memory>
#include <thread>
#include <future>

namespace grf {

class ForestPredictor {
public:
  ForestPredictor(uint num_threads,
                  std::unique_ptr<DefaultPredictionStrategy> strategy);

  ForestPredictor(uint num_threads,
                  std::unique_ptr<OptimizedPredictionStrategy> strategy);

  std::vector<Prediction> predict(const Forest& forest,
                                  const Data& train_data,
                                  const Data& data,
                                  bool estimate_variance) const;

  std::vector<Prediction> predict_oob(const Forest& forest,
                                      const Data& data,
                                      bool estimate_variance) const;

private:
  std::vector<Prediction> predict(const Forest& forest,
                                  const Data& train_data,
                                  const Data& data,
                                  bool estimate_variance,
                                  bool oob_prediction) const;

private:
  TreeTraverser tree_traverser;
  std::unique_ptr<PredictionCollector> prediction_collector;
};

} // namespace grf

#endif //GRF_FORESTPREDICTOR_H
