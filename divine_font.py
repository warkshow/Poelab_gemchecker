import requests
import os
import numpy as np
import itertools
from tabulate import tabulate
from tkinter import *
import screenshot
import time
import cv2
import scipy.signal
import matplotlib.pyplot as plt
import keyboard
import skimage.metrics


def read_gems(path):
    f = open(path, "r")
    gems = f.read()
    return gems.split(",")


class Gem:
    def __init__(self, name, level, quality, color, value, is_transfigured, is_corrupted, count, gem_collection):
        self.name = name
        self.level = level
        self.quality = quality
        self.color = color
        self.value = value
        self.is_transfigured = is_transfigured
        self.is_corrupted = is_corrupted
        self.gem_collection = gem_collection
        self.count = count
    
    def get_transfigured_versions(self):
        transfigured_versions = []
        for gem in self.gem_collection.gems[self.color]:
            if gem.is_transfigured and self.name == gem.name[:len(self.name)] and gem.level == self.level and gem.quality == self.quality: 
                transfigured_versions.append(gem)
        return transfigured_versions



class GemCollection:
    def __init__(self):
        self.gems = {"red": [], "blue": [], "green": []}
        self.colors = ["red", "blue", "green"]
        self.buying_ratio = 1.2 # How much more are you paying than the listed poe.ninja price? default = 1.2 (20% more)
        self.selling_ratio = 0.8 # How much less are you selling for than the listed poe.ninja price? default = 0.8 (20% less)
        self.selling_threshold = 1  # If you buy a gem for x, then whats the lowest value of x * selling_threshold 
                                      #  that you are willing to sell for? default = 2, 1 means you will sell anything 
        self.buying_price_source_amount = 12 # How many lowest price gems of a certain color do you base the buying price of that color on? default = 12
        self.total_change_threshold = 1000 # The total change threshold, if the price has changed by more than total_change_threshold% recently than the gem 
                                         #  is excluded because of likely inaccurate value / pricefixing etc, default = 500
        self.minimum_num_of_listings = 2 # Minimum number of listings of a gem to believe the poeninja value, default = 3
        self.amount_of_transfigured_gems = 8 # Amount of best transfigured gems to show, default = 8

        poeninja_state = requests.get("https://poe.ninja/api/data/index-state").json()
        league_name = poeninja_state["economyLeagues"][0]["name"]
        print(league_name)
        self.gem_prices = requests.get(f"https://poe.ninja/api/data/itemoverview?league={league_name}&type=SkillGem").json()

        self.gem_transfigured_names = {
            "red": read_gems("transfigured_gems_red.txt"), 
            "blue": read_gems("transfigured_gems_blue.txt"), 
            "green": read_gems("transfigured_gems_green.txt")
        }

        self.gem_base_names = {
            "red": read_gems("base_gems_red.txt"), 
            "blue": read_gems("base_gems_blue.txt"), 
            "green": read_gems("base_gems_green.txt")
        }


    def add_gems_with_poeninja(self, gem_names, property_requirements): 
        added_gems = {"red": [], "blue": [], "green": []}
        for color in self.colors:
            for gem in self.gem_prices["lines"]:
                if gem["name"] not in gem_names[color]:
                    continue
                
                gem_meets_requirements = True
                is_transfigured = False

                for (key, value, must_appear) in property_requirements:
                    if must_appear:
                        if key not in gem:
                            #print(f"{key} NOT IN GEM")
                            gem_meets_requirements = False
                            break
                        if gem[key] != value:
                            #print(f"{key} value {gem[key]} DOESNT MATCH {value}")
                            gem_meets_requirements = False
                            break
                    else:
                        if key in gem and gem[key] != value:
                            #print(f"{key} value {gem[key]} DOESNT MATCH {value}")
                            gem_meets_requirements = False
                            break
                
                if gem["name"] in self.gem_transfigured_names[color]:
                    is_transfigured = True

                is_corrupted = False if "corrupted" not in gem else gem["corrupted"]
                quality = 0 if "gemQuality" not in gem else gem["gemQuality"]

                is_low_confidence = abs(gem["lowConfidenceSparkline"]["totalChange"]) > self.total_change_threshold or gem["count"] <= self.minimum_num_of_listings

                if gem_meets_requirements and not is_low_confidence: 
                    new_gem = Gem(gem["name"], gem["gemLevel"], quality, color, gem["chaosValue"], is_transfigured, is_corrupted, gem["count"], self)
                    if "Mirror" in new_gem.name and "Bombarding" in new_gem.name:
                        print((new_gem.name, new_gem.level, new_gem.quality, new_gem.is_corrupted))

                    added_gems[color].append(new_gem)
                else:
                    new_gem = Gem(gem["name"], gem["gemLevel"], quality, color, gem["chaosValue"], is_transfigured, is_corrupted, gem["count"], self)
                    if "Mirror" in new_gem.name and "Bombarding" in new_gem.name:
                        print((new_gem.name, new_gem.level, new_gem.quality, new_gem.is_corrupted, gem_meets_requirements, is_low_confidence, gem["lowConfidenceSparkline"]["totalChange"] ))

        values_by_color = {"red": [], "blue": [], "green": []}
        for color in self.colors:
            for gem in added_gems[color]:
                    values_by_color[color].append(gem.value)
        exists_color = [x for x in added_gems.keys() if len(added_gems[x]) > 0]
        if len(exists_color) != 0:
            level = added_gems[exists_color[0]][0].level
            quality = added_gems[exists_color[0]][0].quality
            is_corrupted = added_gems[exists_color[0]][0].is_corrupted
            is_transfigured = added_gems[exists_color[0]][0].is_transfigured

            for color in self.colors:
                for gem_name in gem_names[color]:
                    if gem_name not in [x.name for x in added_gems[color]]:
                        new_gem = Gem(gem_name, level, quality, color, np.average(values_by_color[color][-self.buying_price_source_amount:]), is_transfigured, is_corrupted, 0, self)
                        added_gems[color].append(new_gem)

            for color in self.colors:
                self.gems[color].extend(added_gems[color])



    def get_best_transform_by_color(self, level, quality, is_transfigured, is_corrupted): 
        values_by_color = {"red": [], "blue": [], "green": []}
        for color in self.colors:
            for gem in self.gems[color]:
                if gem.level == level and gem.quality == quality and gem.is_transfigured == is_transfigured and gem.is_corrupted == is_corrupted:
                    values_by_color[color].append(gem.value)

        price_by_color = {"red": [], "blue": [], "green": []}
        for color in self.colors:
            price_by_color[color] = np.average(values_by_color[color][-self.buying_price_source_amount:]) * self.buying_ratio
        yield_by_color = {"red": [], "blue": [], "green": []}
        for color in self.colors:
            permutations_without_replacement = list(itertools.combinations(values_by_color[color], 3))
            best_gem_choices = [gem_options[0] for gem_options in permutations_without_replacement] # This only works if gems were added through poeninja,
                                                                                                    #  because poeninja gems are sorted based on price.
                                                                                                    # If this assumption is not true then one can check the price of every gem and choose the highest manually 
            yield_by_color[color] = np.average([gem_value * self.selling_ratio if gem_value * self.selling_ratio > price_by_color[color] * self.selling_threshold else price_by_color[color] * self.buying_ratio for gem_value in best_gem_choices])

        profit_by_color = {color: yield_by_color[color] - price_by_color[color] for color in self.colors}

        table = [(color, profit_by_color[color], price_by_color[color], yield_by_color[color]) for color in self.colors]
        table = sorted(table, key = lambda k: k[1])
        table.reverse()


        print(f"Transformation of {level:>2}/{quality:>2} {'corrupted' if is_corrupted else 'non-corrupted'} {'transfigured' if is_transfigured else 'non-transfigured'} gems by gem color.")
        print(tabulate(table, headers=["Color", "Average profit (c)", "Price (c)", "Average yield (c)"]))
        print("\n")


    def get_best_transfigure_by_gem(self, level, quality):
        best_gems = []
        for color in self.colors:
            for gem in self.gems[color]:
                if gem.level == level and gem.quality == quality and not gem.is_transfigured and not gem.is_corrupted:
                    transfigured_options = gem.get_transfigured_versions()
                    if len(transfigured_options) == 0:
                        continue
                    value = np.average([tgem.value for tgem in transfigured_options])
                    best_gems.append((gem.name, gem.value, value, len(transfigured_options)))
        
        best_gems.sort(key=lambda g: g[2] * self.selling_ratio - g[1] * self.buying_ratio)
        best_gems.reverse()
        print(f"Best {level}/{quality} gems to transfigure")
        for (name, price, value, options) in best_gems[:self.amount_of_transfigured_gems]:
            print(f"{name:<18} - avg profit: {value * self.selling_ratio - price * self.buying_ratio:>6.2f}c - price: {price * self.buying_ratio:>6.2f}c - avg sell: {value * self.selling_ratio:>6.2f}c - options: {options:>3}")
        print("\n")

    def get_best_gem_option(self, level, quality, is_corrupted, gem_names):
        gem_options = []
        for gem_name in gem_names:
            found = False
            for color in self.colors:
                for gem in self.gems[color]:
                    if gem.name == gem_name and gem.level == level and gem.quality == quality and gem.is_corrupted == is_corrupted:
                        gem_options.append(gem)
                        found = True
            if not found:
                print(f"{gem_name} {level}/{quality} {'corrupted' if is_corrupted else 'uncorrupted'} was not found on poe.ninja.")
                gem_options.append(Gem(gem_name, level, quality, None, 0, None, is_corrupted, 0, self))
                
        
        table = [(i + 1, gem.name, gem.value, gem.count) for (i, gem) in enumerate(gem_options)]
        sorted_table = sorted(table, key = lambda k: k[2])
        sorted_table.reverse()
        print(f"\nGem detection {level}/{quality} {'corrupted' if is_corrupted else 'uncorrupted'}.")
        print(tabulate(table, headers=["Option", "Gem name", "Yield (c)", "Listings (approx)"]))
        print("")
        print(f"Best is option {table.index(sorted_table[0]) + 1}, {sorted_table[0][1]} at {sorted_table[0][2]}c.\n")


    def get_best_corrupt_transfigure(self, gem_level, gem_quality):
        gem_options = []
        for color in self.colors:
            for gem in self.gems[color]:
                if gem_level == gem.level and gem_quality == gem.quality:
                    if gem.is_transfigured == False:
                        gem_options.append(gem)
        gem_values = []
        for gem in gem_options: 
            if np.isnan(np.average([x.value if not np.isnan(x.value) else 0.0 for x in gem.get_transfigured_versions()])):
                continue
            gem_values.append((gem, np.average([x.value if not np.isnan(x.value) else 0.0 for x in gem.get_transfigured_versions()]) - gem.value, len(gem.get_transfigured_versions())))
        gem_values_table = [(gem.name, value, gem.level, gem.quality, gem.count, options) for (i, (gem, value, options)) in enumerate(gem_values)]
        gem_values_sorted = sorted(gem_values_table, key = lambda k: k[1])
        gem_values_sorted.reverse()
        print(f"Best {gem_level}/{gem_quality} gems to do guaranteed transfigure trick with")
        print(tabulate(gem_values_sorted[:10], headers=["Gem name", "Yield (c)", "Gem level", "Gem quality", "Listings (approx)", "Options"]))
        print("\n")



    def list_best_strategies(self):
        os.system("cls")
        self.get_best_transform_by_color(21, 23, True, True)
        self.get_best_transform_by_color(21, 23, False, True)
        self.get_best_transform_by_color(20, 20, False, False)
        self.get_best_transform_by_color(1, 0, False, False)
        self.get_best_transfigure_by_gem(20, 20)
        self.get_best_transfigure_by_gem(1,0)
        self.get_best_corrupt_transfigure(21, 20)
        self.get_best_corrupt_transfigure(20, 23)
        self.get_best_corrupt_transfigure(21, 23)


def extract_gem_images(img):
    size = 53
    height, width, channels = img.shape
    gem_images = [img[int(32/116*height):int(32/116*height + size/116*height), int(45/342*width + 100/342*width*i):int(45/342*width + 100/342*width*i + size/342*width)] for i in range(3)]
    return gem_images

def alphaMerge(small_foreground, background, top, left):
    """
    Puts a small BGRA picture in front of a larger BGR background.
    :param small_foreground: The overlay image. Must have 4 channels.
    :param background: The background. Must have 3 channels.
    :param top: Y position where to put the overlay.
    :param left: X position where to put the overlay.
    :return: a copy of the background with the overlay added.
    """
    result = background.copy()
    # From everything I read so far, it seems we need the alpha channel separately
    # so let's split the overlay image into its individual channels
    fg_b, fg_g, fg_r, fg_a = cv2.split(small_foreground)
    # Make the range 0...1 instead of 0...255
    fg_a = fg_a / 255.0
    # Multiply the RGB channels with the alpha channel
    label_rgb = cv2.merge([fg_b * fg_a, fg_g * fg_a, fg_r * fg_a])

    # Work on a part of the background only
    height, width = small_foreground.shape[0], small_foreground.shape[1]
    part_of_bg = result[top:top + height, left:left + width, :]
    # Same procedure as before: split the individual channels
    bg_b, bg_g, bg_r = cv2.split(part_of_bg)
    # Merge them back with opposite of the alpha channel
    part_of_bg = cv2.merge([bg_b * (1 - fg_a), bg_g * (1 - fg_a), bg_r * (1 - fg_a)])

    # Add the label and the part of the background
    cv2.add(label_rgb, part_of_bg, part_of_bg)
    # Replace a part of the background
    result[top:top + height, left:left + width, :] = part_of_bg
    return result

def find_divine_font_gems(img, gem_collection, is_transfigured):
    divine_font_gems = extract_gem_images(img)
    gem_images = []
    gem_names = []
    if is_transfigured:
        gem_images = [(cv2.imread(f"gem_transfigured_icons\\{gem_name}.jpg", cv2.IMREAD_UNCHANGED), gem_name) for gem_name in sum(gem_collection.gem_transfigured_names.values(), []) if "Maels" not in gem_name]
    if not is_transfigured:
        gem_images = gem_images + [(cv2.imread(f"gem_base_icons\\{gem_name}.png", cv2.IMREAD_UNCHANGED), gem_name) for gem_name in sum(gem_collection.gem_base_names.values(), [])]
    #gem_images = [(cv2.imread(f"gem_transfigured_icons\{gem_name}.jpg", cv2.IMREAD_UNCHANGED), gem_name) for gem_name in gem_names]
    
    #a = len(gem_images)
    #print(a)
    #gem_images = gem_images + [(cv2.imread(f"gem_base_icons\{gem_name}.png", cv2.IMREAD_UNCHANGED), gem_name) for gem_name in sum(gem_collection.gem_base_names.values(), [])]
    #print(len(base_gem_images))
    #gem_images.extend(base_gem_images)
    #print("________________")
    #print(gem_images[0][0].shape)
    #print(gem_images[0][a + 10].shape)
    #print("________________")
    #print("________________")
    #print(gem_images[185][0].shape)
    #print("________________")
    shape = gem_images[0][0].shape[0:2]
    divine_font_gem_background = cv2.resize(cv2.imread("gem_background.png"), gem_images[0][0].shape[0:2])
    gem_images = [(cv2.resize(gem_image, gem_images[0][0].shape[0:2]), gem_name) for (gem_image, gem_name) in gem_images]
    gem_images = [(alphaMerge(gem_image, divine_font_gem_background, 0, 0), gem_name) for (gem_image, gem_name) in gem_images]

    
    divine_font_gems = [cv2.resize(divine_font_gem, shape) for divine_font_gem in divine_font_gems]
    foundGems = []
    for i, divine_font_gem in enumerate(divine_font_gems):
        closest = (None, 0)
        for gem_image, gem_name in gem_images:
            gem_image_gray = np.sum(gem_image.astype('float'), axis=2)
            divine_font_gem_gray = np.sum(divine_font_gem.astype('float'), axis=2)

            gem_image_gray -= np.mean(gem_image_gray)
            divine_font_gem_gray -= np.mean(divine_font_gem_gray)

            corr_img = scipy.signal.fftconvolve(gem_image_gray, divine_font_gem_gray[::-1,::-1], mode='same')
            offset = np.array(gem_image.shape[0:2]) / 2 - np.unravel_index(np.argmax(corr_img), corr_img.shape) 
            translated_gem_image = cv2.warpAffine(gem_image, np.array([[1.,0.,offset[1]], [0., 1., offset[0]]]), (gem_image.shape[0], gem_image.shape[1]))

            mssim, S = skimage.metrics.structural_similarity(
                divine_font_gem, 
                translated_gem_image, 
                channel_axis=2, 
                data_range=255, 
                gaussian_weights=True, 
                sigma=1.5,
                use_sample_covariance=False,
                full=True)
            
            if mssim > closest[1]:
                closest = (gem_name, mssim)
            
        foundGems.append(closest)
    return foundGems


def divine_font_price_check(*x):
    gem_collection = x[1]
    gem_collection.list_best_strategies()
    

    screenshot.screenshot()
    time.sleep(1)

    img = cv2.imread("divine_font_gems.png")
    divine_font_gems = []
    if x[0] == 2 or x[0] == 1:
        divine_font_gems = find_divine_font_gems(img, gem_collection, True)
    if x[0] == 3:
        divine_font_gems = find_divine_font_gems(img, gem_collection, False)
    divine_font_gems = [x[0] for x in divine_font_gems]
    if x[0] == 1:
        gem_collection.get_best_gem_option(20, 20, False, divine_font_gems)
    if x[0] == 2 or x[0] == 3:
        gem_collection.get_best_gem_option(21, 23, True, divine_font_gems)
    if x[0] == 4:
        gem_collection.get_best_gem_option(1, 0, False, divine_font_gems)




if __name__ == "__main__":
    gem_collection = GemCollection()

    gem_collection.add_gems_with_poeninja(gem_collection.gem_transfigured_names, [("gemLevel", 21, True), ("gemQuality", 23, True)])
    gem_collection.add_gems_with_poeninja(gem_collection.gem_transfigured_names, [("gemLevel", 21, True), ("gemQuality", 20, True)])
    gem_collection.add_gems_with_poeninja(gem_collection.gem_transfigured_names, [("gemLevel", 20, True), ("gemQuality", 23, True)])
    gem_collection.add_gems_with_poeninja(gem_collection.gem_transfigured_names, [("gemLevel", 20, True), ("gemQuality", 20, True), ("corrupted", False, False)])
    gem_collection.add_gems_with_poeninja(gem_collection.gem_base_names, [("gemLevel", 21, True), ("gemQuality", 23, True)])
    gem_collection.add_gems_with_poeninja(gem_collection.gem_base_names, [("gemLevel", 21, True), ("gemQuality", 20, True)])
    gem_collection.add_gems_with_poeninja(gem_collection.gem_base_names, [("gemLevel", 20, True), ("gemQuality", 23, True)])
    gem_collection.add_gems_with_poeninja(gem_collection.gem_base_names, [("gemLevel", 20, True), ("gemQuality", 20, False), ("corrupted", False, False)])
    gem_collection.add_gems_with_poeninja(gem_collection.gem_transfigured_names, [("gemLevel", 1, True), ("gemQuality", 0, False), ("corrupted", False, False)])
    gem_collection.add_gems_with_poeninja(gem_collection.gem_base_names, [("gemLevel", 1, True), ("gemQuality", 0, False), ("corrupted", False, False)])
    gem_collection.list_best_strategies()

    keyboard.add_hotkey('f3', divine_font_price_check, args=[1, gem_collection]) # 20/20 transfigured, uber lab / gift 
    keyboard.add_hotkey('f4', divine_font_price_check, args=[2, gem_collection]) # 21/23 transfigured, dedication
    keyboard.add_hotkey('f5', divine_font_price_check, args=[3, gem_collection]) # 21/23 non-transfigured, dedication
    keyboard.add_hotkey('f6', divine_font_price_check, args=[4, gem_collection]) # 1/0 non-transfigured, uber lab / gift
    keyboard.wait()

"""
gem_collection = GemCollection()

gem_collection.add_gems_with_poeninja(gem_collection.gem_transfigured_names, [("gemLevel", 21, True), ("gemQuality", 23, True)])
gem_collection.add_gems_with_poeninja(gem_collection.gem_transfigured_names, [("gemLevel", 20, True), ("gemQuality", 20, True), ("corrupted", False, False)])
gem_collection.add_gems_with_poeninja(gem_collection.gem_base_names, [("gemLevel", 21, True), ("gemQuality", 23, True)])
gem_collection.add_gems_with_poeninja(gem_collection.gem_base_names, [("gemLevel", 20, True), ("gemQuality", 20, True), ("corrupted", False, False)])
gem_collection.list_best_strategies()

"""