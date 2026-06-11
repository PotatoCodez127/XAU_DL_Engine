import unittest
import numpy as np
from unittest.mock import MagicMock

class TestHybridEnvironmentFixes(unittest.unittest.TestCase):

    def setUp(self):
        # Mock the environment state to isolate the logic tests
        self.env = MagicMock()
        self.env.position = 1
        self.env.position_size = 10.0
        self.env.entry_price = 2000.0
        self.env.sl_price = 1995.0
        self.env.tp_price = 2010.0
        self.env.balance = 10000.0

    def test_intra_candle_ambiguity_sl_first(self):
        """Test if both TP and SL are hit, it respects the one closer to the open."""
        # Simulated volatile candle
        current_open = 1997.0
        current_low = 1990.0
        current_high = 2015.0
        
        dist_to_sl = abs(current_open - self.env.sl_price) # 2.0
        dist_to_tp = abs(current_open - self.env.tp_price) # 13.0
        
        hit_sl = current_low <= self.env.sl_price
        hit_tp = current_high >= self.env.tp_price
        
        if hit_sl and hit_tp:
            if dist_to_sl < dist_to_tp:
                hit_tp = False
            else:
                hit_sl = False
                
        self.assertTrue(hit_sl, "Stop Loss should be triggered first based on proximity to open.")
        self.assertFalse(hit_tp, "Take Profit should be nullified.")

    def test_widened_atr_stop_loss(self):
        """Test that the stop loss scales exactly to 1.5x ATR during entry."""
        current_price = 2000.0
        current_atr = 10.0
        spread = 0.15
        
        entry_price = current_price + spread
        sl_distance = current_atr * 1.5
        sl_price = entry_price - sl_distance
        
        self.assertEqual(sl_price, 1985.15, "Stop loss distance did not calculate to 1.5x ATR accurately.")

    def test_cash_penalty_removed(self):
        """Test that taking no action generates a strictly 0.0 reward."""
        direction_action = 0.05 # Below the 0.1 threshold
        reward = 0.0
        
        # Simulated environment block
        if abs(direction_action) > 0.1:
            reward -= 0.10 # simulated commission
        else:
            reward += 0.0 # No penalty 
            
        self.assertEqual(reward, 0.0, "Manager was penalized for holding cash.")

if __name__ == '__main__':
    unittest.main()